import torch
import torch.nn as nn

from utils.bpe_factory import BPE

class RerankingReducer(nn.Module):
    def __init__(self, config, device, top1=True):
        super(RerankingReducer, self).__init__()
        self.config = config
        self.device = device

        self.top1 = top1

        if config.data.get('reranker', None) is None:
            self.strategy = None
        else:
            self.strategy = config.reranker.strategy

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, presorted=True):
        curr_batch_size = batch[[k for k in batch.keys()][0]].size()[0]
        
        # Pass-through mode: take the top-1 from a pre-sorted set of candidates (eg beam search)
        if self.top1 and presorted:
            output = candidates[:, 0, :]
            output_lens = lengths[:, 0]

            return output, output_lens, scores

        elif self.top1 and self.strategy is None:
            # TODO: sort by score and return top-1
            
            return None

        # TODO: strategic reranking goes here

        return None