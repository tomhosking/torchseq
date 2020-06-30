import torch


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


def get_perplexity(logits, indices, vocab_size=None, ignore_index=None):

    log_probs = torch.sum(torch.softmax(logits, dim=-1) * onehot(indices, vocab_size, ignore_index), dim=-1)

    perplexity = torch.pow(2, -torch.sum(torch.log2(log_probs + 1e-10)))

    return perplexity
