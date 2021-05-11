from torchseq.utils.functions import onehot
import torch

from torchseq.utils.functions import onehot


def get_perplexity(logits, indices, vocab_size=None, ignore_index=None):
    seq_probs = torch.softmax(logits, dim=-1)
    seq_oh = onehot(indices, vocab_size, ignore_index)

    seq_entropy = torch.sum(torch.log2(seq_probs + 1e-10) * seq_oh, dim=-1)

    perplexity = torch.pow(2, -torch.sum(seq_entropy, dim=-1) / seq_oh.sum(dim=-1).sum(dim=-1))

    return perplexity
