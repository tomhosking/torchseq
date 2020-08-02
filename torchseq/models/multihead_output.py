import torch
import torch.nn as nn


class MultiHeadOutput(nn.Module):
    def __init__(
        self,
        embedding_dim,
        vocab_size,
        num_heads=1,
        num_projections=1,
        projection_init=None,
        freeze_projection=False,
        variational=False,
        normed=False,
    ):
        super(MultiHeadOutput, self).__init__()

        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num heads!"

        self.num_heads = num_heads
        self.num_projections = num_projections
        self.dim_per_head = embedding_dim // num_heads
        self.variational = variational

        if num_projections > 1:
            self.embeds_to_logits = nn.ModuleList(
                [nn.Linear(self.dim_per_head, vocab_size, bias=False) for _ in range(num_projections)]
            ).cpu()
        else:
            self.embeds_to_logits = nn.Linear(self.dim_per_head, vocab_size, bias=False).cpu()

        if variational:
            self.embeds_to_logvars = nn.Linear(self.dim_per_head, vocab_size, bias=False).cpu()
            raise Exception("Variation projection is not correct! Need to implement KL loss")

        self.head_weight = nn.Linear(embedding_dim, num_heads * num_projections, bias=False)

        if projection_init is not None:
            self.embeds_to_logits.weight.data = projection_init

        if num_projections > 1:
            for layer in self.embeds_to_logits:
                layer.weight.requires_grad = not freeze_projection
        else:
            self.embeds_to_logits.weight.requires_grad = not freeze_projection

        # if normed:
        #     self.embeds_to_logits = nn.utils.weight_norm(self.embeds_to_logits)

    def forward(self, embeds):
        if self.num_heads > 1 or self.num_projections > 1:
            bsz = embeds.shape[0]
            # Split embeds into num_heads smaller embeddings
            embeds_chunked = embeds.view(bsz, -1, self.num_heads, self.dim_per_head)

            # Project each head
            if self.num_projections > 1:
                with torch.no_grad():
                    for layer in self.embeds_to_logits:
                        layer.weight.div_(torch.norm(layer.weight, dim=1, keepdim=True))
                logits_split = torch.cat([layer(embeds_chunked) for layer in self.embeds_to_logits], dim=2)
            else:
                with torch.no_grad():
                    self.embeds_to_logits.weight.div_(torch.norm(self.embeds_to_logits.weight, dim=1, keepdim=True))
                logits_split = self.embeds_to_logits(embeds_chunked)

            logits_weights = torch.softmax(self.head_weight(embeds), dim=-1).unsqueeze(-1)

            if self.variational:
                mu = logits_split
                logvar = self.embeds_to_logvars(embeds_chunked).unsqueeze(1)

                def reparameterize(mu, logvar):
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + eps * std

                print("var")
                logits_split = reparameterize(mu, logvar)

            # Combine logits from each head
            logits = torch.sum(logits_split * logits_weights, dim=2)

            # logits = torch.log(logits)
        else:
            logits = self.embeds_to_logits(embeds)

            if self.variational:
                mu = logits
                logvar = self.embeds_to_logvars(embeds)

                def reparameterize(mu, logvar):
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + eps * std

                logits = reparameterize(mu, logvar)

        if self.variational:
            return logits
        else:
            return logits
