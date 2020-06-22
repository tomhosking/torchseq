import torch
import torch.nn as nn

from torchseq.utils.tokenizer import Tokenizer

# Get a cross-entropy style loss that penalises any token from a given sequence


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


class SuppressionLoss(nn.Module):
    def __init__(self, config):
        super(SuppressionLoss, self).__init__()
        self.config = config

    def forward(self, logits, penalty_sequence):

        penalty_onehot = onehot(penalty_sequence, N=self.config.prepro.vocab_size, ignore_index=Tokenizer().pad_id)

        penalty_mask = penalty_onehot.sum(dim=-2, keepdim=True)
        penalty_mask = torch.min(penalty_mask, torch.ones_like(penalty_mask))
        probs = nn.functional.softmax(logits, dim=-1)

        loss = penalty_mask * probs

        return loss.sum(dim=-1)
