import torch
import torch.nn as nn


class MultiHeadOutput(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_heads=1, projection_init=None, freeze_projection=False, variational=False):
        super(MultiHeadOutput, self).__init__()

        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num heads!"

        self.num_heads = num_heads
        self.dim_per_head = embedding_dim // num_heads
        self.variational = variational

        self.embeds_to_logits = nn.Linear(self.dim_per_head, vocab_size, bias=False).cpu()
        

        if variational:
            self.embeds_to_logvars = nn.Linear(self.dim_per_head, vocab_size, bias=False).cpu()
            print("var setup")

        self.head_weight = nn.Linear(embedding_dim, num_heads, bias=False).cpu()

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

            logits_weights = torch.softmax(self.head_weight(embeds), dim=-1).unsqueeze(-1)

            if self.variational:
                self.mu = logits_split
                self.logvar = self.embeds_to_logvars(embeds_chunked).unsqueeze(1)

                def reparameterize(mu, logvar):
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + eps * std
                print("var")
                logits_split = reparameterize(self.mu, self.logvar)

            # Combine logits from each head
            logits = torch.sum(logits_split * logits_weights, dim=2)
        else:
            logits = self.embeds_to_logits(embeds)

            if self.variational:
                self.mu = logits
                self.logvar = self.embeds_to_logvars(embeds)

                def reparameterize(mu, logvar):
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + eps * std
                
                logits = reparameterize(self.mu, self.logvar)

            

        return logits
