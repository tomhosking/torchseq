import torch
import torch.nn as nn

from utils.tokenizer import BPE


class NgramReranker(nn.Module):
    def __init__(self, config, device, src_field):
        super(NgramReranker, self).__init__()
        self.config = config
        self.device = device

        self.src_field = src_field

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, presorted=True, top1=True):

        sorted_scores, sorted_indices = torch.sort(scores, descending=True)

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
