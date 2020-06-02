import torch
import torch.nn as nn


class MultiHeadOutput(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_heads=1, projection_init=None, freeze_projection=False):
        super(MultiHeadOutput, self).__init__()

        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num heads!"

        self.num_heads = num_heads
        self.dim_per_head = embedding_dim // num_heads

        self.embeds_to_logits = nn.Linear(self.dim_per_head, vocab_size, bias=False).cpu()

        if projection_init is not None:
            self.embeds_to_logits.weight.data = projection_init

        self.embeds_to_logits.weight.requires_grad = not freeze_projection

    def forward(self, embeds):
        if self.num_heads > 1:
            bsz = embeds.shape[0]
            # Split embeds into num_heads smaller embeddings
            embeds_chunked = embeds.view(bsz, -1, self.num_heads, self.dim_per_head)
            # Project each head
            logits_split = self.embeds_to_logits(embeds_chunked)
            # Combine logits from each head
            logits = torch.sum(logits_split, dim=2)
        else:
            logits = self.embeds_to_logits(embeds)

        return logits
