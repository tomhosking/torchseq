import torch
import torch.nn as nn

from torchseq.utils.tokenizer import Tokenizer, FAIRSEQ_LANGUAGE_CODES
from torchseq.utils.functions import onehot


class BeamSearchSampler(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(BeamSearchSampler, self).__init__()
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

    def forward(self, model, batch, tgt_field):
        curr_batch_size = batch[[k for k in batch.keys() if k[-5:] != "_text"][0]].size()[0]
        max_output_len = self.config.eval.data.get("max_out_len", 32)

        beam_width = self.config.beam_search.beam_width  # number of total hypotheses to maintain
        beam_expansion = (
            self.config.beam_search.beam_expansion
        )  # number of new predictions to add to each hypothesis each step

        prevent_repetition = (
            self.config.beam_search.prevent_repetition
            if "prevent_repetition" in self.config.beam_search.data
            else True
        )

        BART_HACK = self.config.eval.data.get("prepend_eos", False)
        MBART_HACK = self.config.eval.data.get("prepend_langcode", False)

        # Create vector of SOS + placeholder for first prediction
        output_seq = torch.LongTensor(curr_batch_size, beam_width, 1).fill_(self.tokenizer.bos_id).to(self.device)
        scores = torch.FloatTensor(curr_batch_size, beam_width, 1).fill_(0).to(self.device)

        if BART_HACK:
            dummy_token = torch.LongTensor(curr_batch_size, beam_width, 1).fill_(self.tokenizer.eos_id).to(self.device)
            output_seq = torch.cat([dummy_token, output_seq], dim=-1)
            scores = torch.cat([scores, scores], dim=-1)

        if MBART_HACK:
            # lang_token = torch.LongTensor(curr_batch_size, beam_width, 1).fill_(batch["tgt_lang"][0]).to(self.device)
            lang_token = batch["tgt_lang"].unsqueeze(-1).unsqueeze(-1).expand(-1, beam_width, -1)
            eos_token = torch.LongTensor(curr_batch_size, beam_width, 1).fill_(self.tokenizer.eos_id).to(self.device)
            output_seq = torch.cat([eos_token, lang_token], dim=-1)
            scores = torch.cat([scores, scores], dim=-1)

        output_done = torch.BoolTensor(curr_batch_size, beam_width).fill_(False).to(self.device)
        padding = torch.LongTensor(curr_batch_size, beam_width).fill_(self.tokenizer.pad_id).to(self.device)
        pad_probs = (
            torch.FloatTensor(
                curr_batch_size, beam_width, self.config.prepro.get_first(["output_vocab_size", "vocab_size"])
            )
            .fill_(float("-inf"))
            .to(self.device)
        )
        pad_probs[:, :, self.tokenizer.pad_id] = float("0")

        def _tile_batch(x):
            return x.repeat_interleave(beam_width, dim=0)

        batch_tiled = {k: (_tile_batch(x) if k[-5:] != "_text" and k[0] != "_" else x) for k, x in batch.items()}

        seq_ix = 0
        memory = {}
        while torch.sum(output_done) < curr_batch_size * beam_width and seq_ix < max_output_len:

            new_logits, memory = model(batch_tiled, output_seq.view(curr_batch_size * beam_width, -1), memory)
            new_logits = new_logits.view(
                curr_batch_size,
                beam_width,
                -1,
                self.config.prepro.get_first(["output_vocab_size", "vocab_size"]),
            )
            output_done = (output_seq[:, :, -1] == self.tokenizer.pad_id) | (
                output_seq[:, :, -1] == self.tokenizer.eos_id
            )

            if prevent_repetition:
                one_hot_prev = onehot(
                    output_seq[:, :, -1], N=self.config.prepro.get_first(["output_vocab_size", "vocab_size"])
                )
                new_logits[:, :, -1, :] = new_logits[:, :, -1, :] + (one_hot_prev * float("-1e-16"))

            new_probs = torch.where(
                output_done.unsqueeze(-1), pad_probs, torch.log_softmax(new_logits[:, :, -1, :], -1)
            )

            if seq_ix == 0:
                top_expansions = torch.topk(new_probs, k=beam_width, dim=-1, largest=True)

                # On first iteration, the beams are all the same! So spread the topk across beams
                output_seq = torch.cat(
                    [output_seq, top_expansions.indices.unsqueeze(2)[:, 0, :, :].permute(0, 2, 1)], dim=-1
                )
                scores = torch.cat([scores, top_expansions.values.unsqueeze(2)[:, 0, :, :].permute(0, 2, 1)], dim=-1)
                # print(scores)
                # exit()
            else:

                top_expansions = torch.topk(new_probs, k=beam_expansion, dim=-1, largest=True)

                expanded_beam_ixs = torch.cat(
                    [
                        output_seq.unsqueeze(-2).expand(-1, -1, beam_expansion, -1),
                        top_expansions.indices.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                expanded_beam_scores = torch.cat(
                    [scores.unsqueeze(-2).expand(-1, -1, beam_expansion, -1), top_expansions.values.unsqueeze(-1)],
                    dim=-1,
                )

                curr_seq_len = expanded_beam_ixs.shape[3]

                expanded_beam_ixs = expanded_beam_ixs.view(curr_batch_size, beam_width * beam_expansion, curr_seq_len)

                hypothesis_len = torch.sum(expanded_beam_ixs != self.tokenizer.pad_id, dim=-1)
                # Length penalty needs to be applied to *overall* score, not score for this token
                len_alpha = self.config.beam_search.length_alpha
                length_penalty = torch.pow((5 + hypothesis_len).float(), len_alpha) / pow(5.0 + 1.0, len_alpha)

                expanded_beam_scores = expanded_beam_scores.view(
                    curr_batch_size, beam_width * beam_expansion, curr_seq_len
                )

                expanded_beam_scores = expanded_beam_scores

                beam_scores = torch.sum(expanded_beam_scores, dim=-1).to(scores) / length_penalty

                top_beams = torch.topk(beam_scores, k=beam_width, dim=-1)

                scores = torch.gather(
                    expanded_beam_scores, 1, top_beams.indices.unsqueeze(-1).expand(-1, -1, curr_seq_len)
                )
                new_output = torch.gather(
                    expanded_beam_ixs, 1, top_beams.indices.unsqueeze(-1).expand(-1, -1, curr_seq_len)
                )

                # Use pad for the output for elements that have completed
                output_done = (new_output[:, :, -2] == self.tokenizer.eos_id) | (
                    new_output[:, :, -2] == self.tokenizer.pad_id
                )
                new_output[:, :, -1] = torch.where(output_done, padding, new_output[:, :, -1])

                output_seq = new_output

            seq_ix += 1

        # Sort by score

        # if BART_HACK or MBART_HACK:
        #     output_seq = output_seq[:, :, 1:]
        #     scores = scores[:, :, 1:]

        output_len = torch.sum(output_seq != self.tokenizer.pad_id, dim=-1)
        length_penalty = torch.pow((5 + output_len).float(), len_alpha) / pow(5.0 + 1.0, len_alpha)
        beam_scores = torch.sum(scores, dim=-1) / length_penalty

        sorted_scores, sorted_indices = torch.sort(beam_scores, descending=True)

        output_seq = torch.gather(output_seq, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, output_seq.shape[2]))

        return output_seq, sorted_scores, output_len, {}
