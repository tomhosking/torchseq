import torch.nn as nn

from datasets.loaders import PAD

class RnnAqModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim,
            padding_idx=0) # TODO: this should come from a config


    def forward(self, batch):
        print(batch['c'])
        return batch['q']
