import torch
import torch.nn as nn

from utils.tokenizer import BPE


def pad_to_match(x1, x2, pad_id):
    l1 = x1.shape[1]
    l2 = x2.shape[1]
    if l1 == l2:
        return x1, x2

    pad_required = max(l1 - l2, l2 - l1)
    pad_toks = torch.full((x1.shape[0], pad_required), pad_id, dtype=x1.dtype, device=x1.device)

    if l1 > l2:
        x2 = torch.cat(x2, pad_toks, dim=1)
    elif l2 > l1:
        x1 = torch.cat(x1, pad_toks, dim=1)

    return x1, x2


class BacktranslateReranker(nn.Module):
    def __init__(self, config, device, src_field, model, decoder, loss):
        super(BacktranslateReranker, self).__init__()
        self.config = config
        self.device = device

        self.src_field = src_field
        self.model = model
        self.decoder = decoder
        self.loss = nn.CrossEntropyLoss(ignore_index=BPE.pad_id, reduction="none")

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, presorted=True, top1=True):

        # Flatten to a single (large) batch
        candidates_flattened = torch.flatten(candidates, 0, 1)
        lengths_flattened = torch.flatten(lengths)

        num_candidates = candidates.shape[1]
        original_length = batch[self.src_field].shape[1]
        tgt_seq_tiled = torch.repeat_interleave(batch[self.src_field], repeats=num_candidates, dim=0)

        # Pad candidates to match  if necessary
        if original_length > candidates_flattened.shape[1]:
            pad_toks = torch.full(
                (tgt_seq_tiled.shape[0], original_length - candidates_flattened.shape[1]),
                BPE.pad_id,
                dtype=tgt_seq_tiled.dtype,
                device=tgt_seq_tiled.device,
            )
            tgt_seq_tiled = torch.cat([tgt_seq_tiled, pad_toks], dim=1)

        # print("orig", batch[self.src_field].shape)
        # print("tiled", tgt_seq_tiled.shape)
        # print("cands", candidates_flattened.shape)

        batch_backtranslate = {
            self.src_field: candidates_flattened,
            self.src_field + "_len": lengths_flattened,
        }

        # print(batch_backtranslate)

        # Get nll of original source using candidates as input
        _, logits, _ = self.decoder(self.model, batch_backtranslate, self.src_field)

        # Truncate logits if the input was longer than tgt
        # TODO: is this right?
        if logits.shape[1] > tgt_seq_tiled.shape[1]:
            logits = logits[:, : tgt_seq_tiled.shape[1]]

        this_loss = self.loss(logits.permute(0, 2, 1), tgt_seq_tiled)
        nlls = torch.sum(this_loss, dim=1) / (batch[self.src_field + "_len"] - 1).to(this_loss)

        # reshape back to beam-wise scores
        scores = nlls.reshape_as(lengths)

        # sort with lowest nll (= highest likelihood) first
        sorted_scores, sorted_indices = torch.sort(scores, descending=False)

        sorted_seqs = torch.gather(candidates, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, candidates.shape[2]))

        if top1:
            output = sorted_seqs[:, 0, :]
        else:
            topk = self.config.eval.data.get("topk", None)
            if topk is not None:
                output = sorted_seqs[:, :topk, :]
            else:
                output = sorted_seqs[:, 0, :]

        return output, torch.sum(output != BPE.pad_id, dim=-1), sorted_scores
