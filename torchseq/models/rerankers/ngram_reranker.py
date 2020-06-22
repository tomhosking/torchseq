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


class NgramReranker(nn.Module):
    def __init__(self, config, device, src_field):
        super(NgramReranker, self).__init__()
        self.config = config
        self.device = device

        self.src_field = src_field

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, sort=True, top1=True):

        # Get k-hot representations of the ref and candidate sequences
        # Also add in the "beam" dimension
        refs_k_hot = (
            torch.sum(
                onehot(batch[self.src_field], N=self.config.prepro.vocab_size, ignore_index=Tokenizer().pad_id), -2
            )
            .float()
            .unsqueeze(1)
        )

        candidates_k_hot = torch.sum(
            onehot(candidates, N=self.config.prepro.vocab_size, ignore_index=Tokenizer().pad_id), -2
        ).float()

        # print(self.src_field, batch[self.src_field].shape)
        # print(candidates.shape)
        # print(refs_k_hot.shape, candidates_k_hot.shape)

        # take dot product to find token overlap between ref and candidates
        scores = torch.matmul(refs_k_hot, candidates_k_hot.transpose(-1, -2))

        # print(scores.shape)
        scores = scores.squeeze(1) / (refs_k_hot.norm(dim=-1) * candidates_k_hot.norm(dim=-1))
        # print(scores.shape)
        # Convert to fraction of different tokens, so that highest is best
        scores = 1 - scores

        if sort:
            scores, sorted_indices = torch.sort(scores, descending=True)

            candidates = torch.gather(candidates, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, candidates.shape[2]))

        if top1:
            output = candidates[:, 0, :]
        else:
            topk = self.config.eval.data.get("topk", None)
            if topk is not None:
                output = candidates[:, :topk, :]
            else:
                output = candidates[:, 0, :]

        return output, torch.sum(output != Tokenizer().pad_id, dim=-1), scores
