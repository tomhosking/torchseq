import torch
import torch.nn as nn

from torchseq.utils.tokenizer import Tokenizer


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


"""
Based on https://arxiv.org/pdf/1610.02424.pdf
When extending each hypothesis, consider the predictions made by previous elements in the beam, and enforce a diversity penalty.
"""


class DiverseBeamSearchSampler(nn.Module):
    def __init__(self, config, device):
        super(DiverseBeamSearchSampler, self).__init__()
        self.config = config
        self.device = device

    def forward(self, model, batch, tgt_field):
        curr_batch_size = batch[[k for k in batch.keys()][0]].size()[0]
        max_output_len = self.config.eval.data.get("max_out_len", 32)

        beam_width = self.config.diverse_beam.beam_width  # number of total hypotheses to maintain
        beam_expansion = (
            self.config.diverse_beam.beam_expansion
        )  # number of new predictions to add to each hypothesis each step

        num_groups = self.config.diverse_beam.num_groups
        penalty_weight = self.config.diverse_beam.penalty_weight

        assert beam_width % num_groups == 0, "Beam width for DBS must be divisible by num groups!"

        prevent_repetition = self.config.diverse_beam.data.get("prevent_repetition", True)

        BART_HACK = self.config.eval.data.get("prepend_eos", False)

        # Create vector of SOS + placeholder for first prediction
        output_seq = torch.LongTensor(curr_batch_size, beam_width, 1).fill_(Tokenizer().bos_id).to(self.device)
        scores = torch.FloatTensor(curr_batch_size, beam_width, 1).fill_(0).to(self.device)

        # seed with an extra eos token to mimic fairseq
        if BART_HACK:
            dummy_token = torch.LongTensor(curr_batch_size, beam_width, 1).fill_(Tokenizer().eos_id).to(self.device)
            output_seq = torch.cat([dummy_token, output_seq], dim=-1)
            scores = torch.cat([scores, scores], dim=-1)

        output_done = torch.BoolTensor(curr_batch_size, beam_width).fill_(False).to(self.device)
        padding = torch.LongTensor(curr_batch_size, beam_width).fill_(Tokenizer().pad_id).to(self.device)
        pad_probs = (
            torch.FloatTensor(curr_batch_size, beam_width, self.config.prepro.vocab_size)
            .fill_(float("-inf"))
            .to(self.device)
        )
        pad_probs[:, :, Tokenizer().pad_id] = float("0")

        def _tile_batch(x):
            return x.repeat_interleave(beam_width, dim=0)

        batch_tiled = {k: _tile_batch(x) for k, x in batch.items() if k[-5:] != "_text"}

        seq_ix = 0
        memory = {}
        while torch.sum(output_done) < curr_batch_size * beam_width and seq_ix < max_output_len:

            new_logits, memory = model(batch_tiled, output_seq.view(curr_batch_size * beam_width, -1), memory)
            new_logits = new_logits.view(curr_batch_size, beam_width, -1, self.config.prepro.vocab_size)
            output_done = (output_seq[:, :, -1] == Tokenizer().pad_id) | (output_seq[:, :, -1] == Tokenizer().eos_id)

            if prevent_repetition:
                one_hot_prev = onehot(output_seq[:, :, -1], N=self.config.prepro.vocab_size)
                new_logits[:, :, -1, :] = new_logits[:, :, -1, :] + (one_hot_prev * float("-1e-16"))

            new_log_probs = torch.where(
                output_done.unsqueeze(-1), pad_probs, torch.log_softmax(new_logits[:, :, -1, :], -1)
            )

            if seq_ix == 0:
                top_expansions = torch.topk(new_log_probs, k=beam_width, dim=-1, largest=True)

                # On first iteration, the beams are all the same! So spread the topk across beams
                output_seq = torch.cat(
                    [output_seq, top_expansions.indices.unsqueeze(2)[:, 0, :, :].permute(0, 2, 1)], dim=-1
                )
                scores = torch.cat([scores, top_expansions.values.unsqueeze(2)[:, 0, :, :].permute(0, 2, 1)], dim=-1)
                # print(scores)
                # exit()
            else:

                scores_by_group = list(scores.chunk(num_groups, dim=1))
                output_by_group = list(output_seq.chunk(num_groups, dim=1))
                new_log_probs_by_group = list(new_log_probs.chunk(num_groups, dim=1))
                new_output = [None] * num_groups

                for gix in range(num_groups):
                    if gix > 0:
                        # augment scores with diversity penalty
                        # which tokens have already been output at this step by other groups
                        used_ids = torch.cat([seq[:, :, -1] for seq in new_output[:gix]], dim=1)

                        used_ids_onehot = onehot(
                            used_ids, N=self.config.prepro.vocab_size, ignore_index=Tokenizer().pad_id
                        )

                        # build a mask of the already used tokens
                        penalty_mask = torch.sum(used_ids_onehot, dim=1, keepdim=True).float()

                        # subtract the penalty from the log probs
                        new_log_probs_by_group[gix] -= penalty_mask * penalty_weight

                    # Generate expanded hypotheses
                    top_expansions = torch.topk(new_log_probs_by_group[gix], k=beam_expansion, dim=-1, largest=True)

                    # Concat with previous seqs
                    expanded_beam_ixs = torch.cat(
                        [
                            output_by_group[gix].unsqueeze(-2).expand(-1, -1, beam_expansion, -1),
                            top_expansions.indices.unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                    expanded_beam_scores = torch.cat(
                        [
                            scores_by_group[gix].unsqueeze(-2).expand(-1, -1, beam_expansion, -1),
                            top_expansions.values.unsqueeze(-1),
                        ],
                        dim=-1,
                    )

                    curr_seq_len = expanded_beam_ixs.shape[3]

                    # Reshape to bsz x (beam*beam) x seq
                    expanded_beam_ixs = expanded_beam_ixs.view(
                        curr_batch_size, beam_width * beam_expansion // num_groups, curr_seq_len
                    )

                    # Calculate length penalty
                    hypothesis_len = torch.sum(expanded_beam_ixs != Tokenizer().pad_id, dim=-1)
                    len_alpha = self.config.diverse_beam.length_alpha
                    length_penalty = torch.pow((5 + hypothesis_len).float(), len_alpha) / pow(5.0 + 1.0, len_alpha)

                    # Find top beams
                    expanded_beam_scores = expanded_beam_scores.view(
                        curr_batch_size, beam_width * beam_expansion // num_groups, curr_seq_len
                    )

                    # Length penalty needs to be applied to *overall* score, not score for this token
                    beam_scores = torch.sum(expanded_beam_scores, dim=-1).to(scores_by_group[gix]) / length_penalty

                    top_beams = torch.topk(beam_scores, k=beam_width // num_groups, dim=-1)

                    # Reduce to just the best hypotheses
                    scores_by_group[gix] = torch.gather(
                        expanded_beam_scores, 1, top_beams.indices.unsqueeze(-1).expand(-1, -1, curr_seq_len)
                    )
                    new_output[gix] = torch.gather(
                        expanded_beam_ixs, 1, top_beams.indices.unsqueeze(-1).expand(-1, -1, curr_seq_len)
                    )

                # recombine the group outputs
                new_output = torch.cat(new_output, dim=1)
                scores = torch.cat(scores_by_group, dim=1)

                # Use pad for the output for elements that have completed
                output_done = (new_output[:, :, -2] == Tokenizer().eos_id) | (
                    new_output[:, :, -2] == Tokenizer().pad_id
                )
                new_output[:, :, -1] = torch.where(output_done, padding, new_output[:, :, -1])

                output_seq = new_output

            seq_ix += 1

        # remove the EOS from the start
        if BART_HACK:
            output_seq = output_seq[:, :, 1:]
            scores = scores[:, :, 1:]

        # apply length penalty
        output_len = torch.sum(output_seq != Tokenizer().pad_id, dim=-1)
        length_penalty = torch.pow((5 + output_len).float(), len_alpha) / pow(5.0 + 1.0, len_alpha)
        beam_scores = torch.sum(scores, dim=-1) / length_penalty

        # Sort by score
        sorted_scores, sorted_indices = torch.sort(beam_scores, descending=True)

        output_seq = torch.gather(output_seq, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, output_seq.shape[2]))

        return output_seq, sorted_scores, output_len, {}
