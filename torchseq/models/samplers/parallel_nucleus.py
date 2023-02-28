import torch
import torch.nn as nn

from torchseq.utils.tokenizer import Tokenizer


from torchseq.utils.functions import onehot, top_k_top_p_filtering


class ParallelNucleusSampler(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(ParallelNucleusSampler, self).__init__()
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

    def forward(self, model, batch, tgt_field):
        curr_batch_size = batch[[k for k in batch.keys() if k[-5:] != "_text"][0]].size()[0]

        max_output_len = self.config.eval.data.get("max_out_len", 32)

        prevent_repetition = (
            self.config.nucleus_sampling.prevent_repetition
            if "prevent_repetition" in self.config.nucleus_sampling.data
            else True
        )

        if not self.config.eval.data.get("shifted_decoding", True):
            print("Unshifted decoding not supported by nucleus decoder!")

        beam_width = self.config.nucleus_sampling.beam_width  # number of total hypotheses to maintain
        prob_cutoff = self.config.nucleus_sampling.cutoff

        # Create vector of SOS + placeholder for first prediction
        output_seq = torch.LongTensor(curr_batch_size, beam_width, 1).fill_(self.tokenizer.bos_id).to(self.device)
        scores = torch.FloatTensor(curr_batch_size, beam_width, 1).fill_(1).to(self.device)

        output_done = torch.BoolTensor(curr_batch_size, beam_width).fill_(False).to(self.device)
        padding = torch.LongTensor(curr_batch_size, beam_width).fill_(self.tokenizer.pad_id).to(self.device)
        pad_probs = (
            torch.FloatTensor(
                curr_batch_size, beam_width, self.config.prepro.get_first(["output_vocab_size", "vocab_size"])
            )
            .fill_(float("0"))
            .to(self.device)
        )
        pad_probs[:, :, self.tokenizer.pad_id] = float("1")

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

            new_logits = top_k_top_p_filtering(logits=new_logits, top_p=prob_cutoff)

            if prevent_repetition:
                one_hot_prev = onehot(
                    output_seq[:, :, -1], N=self.config.prepro.get_first(["output_vocab_size", "vocab_size"])
                )
                new_logits[:, :, -1, :] = new_logits[:, :, -1, :] + (one_hot_prev * float("-1e-16"))

            new_probs = torch.where(
                output_done.unsqueeze(-1), pad_probs, nn.functional.softmax(new_logits[:, :, -1, :], -1)
            )

            sampled_indices = (
                torch.multinomial(new_probs.view(curr_batch_size * beam_width, -1).cpu(), 1)
                .view(curr_batch_size, beam_width, -1)
                .to(self.device)
            )

            sampled_scores = new_probs.gather(index=sampled_indices, dim=-1)

            new_output = torch.cat([output_seq, sampled_indices], dim=-1)
            scores = torch.cat([scores, sampled_scores], dim=-1)

            # Use pad for the output for elements that have completed
            if seq_ix > 0:
                output_done = (new_output[:, :, -2] == self.tokenizer.eos_id) | (
                    new_output[:, :, -2] == self.tokenizer.pad_id
                )
                new_output[:, :, -1] = torch.where(output_done, padding, new_output[:, :, -1])

            output_seq = new_output

            seq_ix += 1

        # Take top-1 beam:
        hypothesis_len = torch.sum(output_seq != self.tokenizer.pad_id, dim=-1)

        # Length penalty needs to be applied to *overall* score, not score for this token
        len_alpha = self.config.nucleus_sampling.length_alpha
        length_penalty = torch.pow((5 + hypothesis_len).float(), len_alpha) / pow(5.0 + 1.0, len_alpha)

        beam_scores = (
            torch.log(scores)
            .where(output_seq != self.tokenizer.pad_id, torch.FloatTensor([0.0]).to(self.device))
            .sum(-1)
            / length_penalty
        )

        sorted_scores, sorted_indices = torch.sort(beam_scores, descending=True)

        output_seq = torch.gather(output_seq, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, output_seq.shape[2]))

        output = output_seq

        return output, sorted_scores, torch.sum(output_seq != self.tokenizer.pad_id, dim=-1), memory
