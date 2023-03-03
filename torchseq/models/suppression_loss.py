import torch
import torch.nn as nn

from torchseq.utils.tokenizer import Tokenizer

# Get a cross-entropy style loss that penalises any token from a given sequence
from torchseq.utils.functions import onehot


class SuppressionLoss(nn.Module):
    def __init__(self, config):
        super(SuppressionLoss, self).__init__()
        self.config = config

    def forward(self, logits, penalty_sequence, pad_id):
        penalty_onehot = onehot(
            penalty_sequence,
            N=self.config.prepro.get_first(["output_vocab_size", "vocab_size"]),
            ignore_index=pad_id,
        )

        penalty_mask = penalty_onehot.sum(dim=-2, keepdim=True)
        penalty_mask = torch.min(penalty_mask, torch.ones_like(penalty_mask))
        probs = nn.functional.softmax(logits, dim=-1)

        loss = penalty_mask * probs

        return loss.sum(dim=-1)
