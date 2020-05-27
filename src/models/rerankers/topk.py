import torch
import torch.nn as nn

from utils.tokenizer import BPE


class TopkReducer(nn.Module):
    def __init__(self, config, device):
        super(TopkReducer, self).__init__()
        self.config = config
        self.device = device

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, sort=True, top1=True):

        # Skip sorting for now - this is unnecessary compute - if a sampling method that does not return sorted output appears this will need to change!
        #  if sort:
        #     # Sort with lowest scores first - we want to minimise overlap
        #     scores, sorted_indices = torch.sort(scores, descending=False)

        #     candidates = torch.gather(candidates, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, candidates.shape[2]))

        # Pass-through mode: take the top-1 from a pre-sorted set of candidates (eg beam search)
        if top1:
            output = candidates[:, 0, :]

            return output, torch.sum(output != BPE.pad_id, dim=-1), scores

        else:
            return candidates, torch.sum(candidates != BPE.pad_id, dim=-1), scores
