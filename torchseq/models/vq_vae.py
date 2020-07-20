import torch
import torch.nn as nn

# https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb


class VectorQuantizerMultiHead(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
        num_heads=1,
        residual=False,
        ema=True,
        code_offset=0,
        num_residual=0,
    ):
        super(VectorQuantizerMultiHead, self).__init__()

        self._num_embeddings = num_embeddings
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim // self._num_heads

        self._ema = ema
        self._code_offset = code_offset
        self._num_residual = num_residual

        self._embedding = nn.ModuleList(
            [nn.Embedding(self._num_embeddings, self._embedding_dim) for _ in range(num_heads)]
        )
        for embedding in self._embedding:
            embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self._ema_w = nn.ParameterList(
            [nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim)) for _ in range(num_heads)]
        )
        for ix in range(self._num_heads):
            self.register_buffer("_ema_cluster_size" + str(ix), torch.zeros(num_embeddings))
            self._ema_w[ix].data.normal_()

        self._decay = decay
        self._epsilon = epsilon
        self._residual = residual
        self._alpha = nn.Parameter(torch.Tensor(num_heads))

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._num_heads, self._embedding_dim)

        quantized_list = []
        vq_codes = []
        for head_ix, embedding in enumerate(self._embedding):

            this_input = flat_input[:, head_ix, :]
            if head_ix >= self._num_residual:

                this_input = flat_input[:, head_ix, :]

                # Calculate distances
                distances = (
                    torch.sum(flat_input[:, head_ix, :] ** 2, dim=1, keepdim=True)
                    + torch.sum(embedding.weight ** 2, dim=1)
                    - 2 * torch.matmul(flat_input[:, head_ix, :], embedding.weight.t())
                )

                # Encoding
                if not isinstance(self._code_offset, int) or self._code_offset > 0:
                    # Allow for nudging the encodings away from nearest
                    this_offset = (
                        self._code_offset[head_ix] if not isinstance(self._code_offset, int) else self._code_offset
                    )
                    min_k = torch.topk(distances, this_offset + 1, dim=1, largest=False).indices
                    encoding_indices = min_k[:, this_offset].unsqueeze(1)
                else:
                    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

                encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
                encodings.scatter_(1, encoding_indices, 1)
                vq_codes.append(encoding_indices)

                # Quantize and unflatten
                this_quantized = torch.matmul(encodings, embedding.weight)
                quantized_list.append(this_quantized)

                # Use EMA to update the embedding vectors
                if self.training and self._ema:
                    _ema_cluster_size = getattr(self, "_ema_cluster_size" + str(head_ix))
                    _ema_cluster_size = _ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

                    # Laplace smoothing of the cluster size
                    n = torch.sum(_ema_cluster_size.data)
                    _ema_cluster_size = (
                        (_ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
                    )
                    setattr(self, "_ema_cluster_size" + str(head_ix), _ema_cluster_size)

                    dw = torch.matmul(encodings.t(), this_input)
                    self._ema_w[head_ix] = nn.Parameter(self._ema_w[head_ix] * self._decay + (1 - self._decay) * dw)

                    self._embedding[head_ix].weight = nn.Parameter(
                        self._ema_w[head_ix] / _ema_cluster_size.unsqueeze(1)
                    )
            else:

                quantized_list.append(this_input)

        quantized = torch.cat(quantized_list, dim=1).view(input_shape)

        # Loss
        if not self._ema:
            q_latent_loss = nn.functional.mse_loss(quantized, inputs.detach())
        else:
            q_latent_loss = 0
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss + q_latent_loss

        # Straight Through Estimator
        if self._residual:
            # Calculate residual weightings
            resid_weight = torch.sigmoid(self._alpha).unsqueeze(-1)

            alpha_loss = torch.sum(torch.square(resid_weight))
            loss += alpha_loss

            quantized = (
                inputs.view(-1, self._num_heads, self._embedding_dim) * resid_weight
                + quantized.detach().view(-1, self._num_heads, self._embedding_dim) * (1 - resid_weight)
            ).view(input_shape)
        else:
            quantized = inputs + (quantized - inputs).detach()

        return loss, quantized, vq_codes


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5, num_heads=1, residual=False
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        q_latent_loss = nn.functional.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return loss, quantized
