import torch
import torch.nn as nn


class TopkReducer(nn.Module):
    def __init__(self, config, device):
        super(TopkReducer, self).__init__()
        self.config = config
        self.device = device

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, presorted=True, top1=True):

        # Pass-through mode: take the top-1 from a pre-sorted set of candidates (eg beam search)
        if top1 and presorted:
            output = candidates[:, 0, :]
            output_lens = lengths[:, 0]

            return output, output_lens, scores

        elif top1:
            # TODO: sort by score and return top-1
            raise NotImplementedError("top-1 filtering with unsorted inputs is not implemented!!!")
            return None
        elif presorted:
            return candidates, lengths, scores

        return candidates, lengths, scores
