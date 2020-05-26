import torch
import torch.nn as nn

from utils.tokenizer import BPE


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


class NgramReranker(nn.Module):
    def __init__(self, config, device, src_field):
        super(NgramReranker, self).__init__()
        self.config = config
        self.device = device

        self.src_field = src_field

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, presorted=True, top1=True):

        # Get k-hot representations of the ref and candidate sequences
        refs_k_hot = torch.sum(
            onehot(batch[self.src_field], N=self.config.prepro.vocab_size, ignore_index=BPE.pad_id), -2, keepdim=True
        ).float()

        candidates_k_hot = torch.sum(
            onehot(candidates, N=self.config.prepro.vocab_size, ignore_index=BPE.pad_id), -2, keepdim=True
        ).float()

        # take dot product to find token overlap between ref and candidates
        scores = torch.matmul(refs_k_hot, candidates_k_hot.transpose(-1, -2))
        scores = scores.squeeze(-1).squeeze(-1) / (
            refs_k_hot.squeeze(-2).norm(dim=-1) * candidates_k_hot.squeeze(-2).norm(dim=-1)
        )

        # Sort with lowest scores first - we want to minimise overlap
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
