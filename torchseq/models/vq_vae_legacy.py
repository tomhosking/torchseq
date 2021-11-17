from torchseq.utils.functions import cos_sim, onehot
import torch
import torch.nn as nn

from math import e, floor

# https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb


class VectorQuantizerMultiHeadLegacy(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
        num_heads=1,
        residual=False,
        residual_head_range=(0, 0),
        ema=True,
        code_offset=0,
        soft_em=True,
        warmup_steps=None,
        code_entropy_weight=0,
        hierarchical=False,
        hierarchical_balance_dims=False,
        hierarchical_greedy=False,
        transitions=False,
        transitions_bias=False,
        transitions_embed=False,
        transitions_log=False,
        use_cosine_similarities=False,
        use_gumbel=False,
        gumbel_temp=1.0,
        use_straight_through=True,
        separate_output_embedding=False,
        use_code_classifier=False,
        additive=False,
        only_final=False,
        subtract_previous=False,
    ):

        # residual_head_range=(0, 0),
        super(VectorQuantizerMultiHeadLegacy, self).__init__()

        self._num_embeddings = num_embeddings
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim // self._num_heads

        self._ema = ema
        self._code_offset = code_offset
        self._residual_head_range = residual_head_range
        self._soft_em = soft_em
        self._warmup_steps = warmup_steps

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
        self._code_entropy_weight = code_entropy_weight

        if code_entropy_weight > 0:
            cooccur_shape = [num_embeddings] * (num_heads - (residual_head_range[1] - residual_head_range[0]))
            self.register_buffer("_code_cooccurrence", torch.zeros(*cooccur_shape))

    def forward(self, inputs, global_step=None, forced_codes=None):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._num_heads, self._embedding_dim)

        if forced_codes is not None and forced_codes.shape[1] == (self._num_heads - self._residual_head_range[1]):
            padding = torch.zeros_like(forced_codes)[:, :1].expand(-1, self._residual_head_range[1])
            forced_codes = torch.cat([padding, forced_codes], dim=1)

        quantized_list = []
        vq_codes = []
        all_encodings = []
        for head_ix, embedding in enumerate(self._embedding):

            this_input = flat_input[:, head_ix, :]
            if head_ix >= self._residual_head_range[0] and head_ix < self._residual_head_range[1]:
                quantized_list.append(this_input)
            else:

                this_input = flat_input[:, head_ix, :]

                # Calculate distances
                distances = (
                    torch.sum(flat_input[:, head_ix, :] ** 2, dim=1, keepdim=True)
                    + torch.sum(embedding.weight ** 2, dim=1)
                    - 2 * torch.matmul(flat_input[:, head_ix, :], embedding.weight.t())
                )

                # Encoding
                if forced_codes is not None:
                    assert (
                        forced_codes.shape[1] == self._num_heads
                    ), "If forced_codes is supplied, it must be the same length as the number of quantizer heads!"
                    encoding_indices = forced_codes[:, head_ix].unsqueeze(1)

                    encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
                    encodings.scatter_(1, encoding_indices, 1)

                elif not isinstance(self._code_offset, int) or self._code_offset > 0:
                    # Allow for nudging the encodings away from nearest
                    this_offset = (
                        self._code_offset[head_ix] if not isinstance(self._code_offset, int) else self._code_offset
                    )
                    min_k = torch.topk(distances, this_offset + 1, dim=1, largest=False).indices
                    encoding_indices = min_k[:, this_offset].unsqueeze(1)

                    encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
                    encodings.scatter_(1, encoding_indices, 1)

                elif self._soft_em and self.training:
                    encodings = torch.softmax(-1.0 * distances, dim=-1).detach()
                    encoding_indices = torch.argmax(encodings, dim=-1)
                else:
                    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

                    encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
                    encodings.scatter_(1, encoding_indices, 1)
                    encoding_indices = encoding_indices.squeeze(1)

                vq_codes.append(encoding_indices)
                all_encodings.append(encodings.unsqueeze(1))

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

        quantized = torch.cat(quantized_list, dim=1).view(input_shape)
        all_encodings = torch.cat(all_encodings, dim=1)

        code_entropy_loss = 0
        if self._code_entropy_weight > 0:

            # h_ix x bsz
            cooccur_ixs = list(zip(*vq_codes))
            cooccur_mask = torch.zeros_like(self._code_cooccurrence)
            for codes in cooccur_ixs:
                cooccur_mask[codes] += 1.0

            bsz = encodings.shape[0]

            # print(self._code_cooccurrence.shape)
            # print(cooccur_mask.shape)
            # print(all_encodings.shape)

            alpha = 1 / 100
            self._code_cooccurrence *= 1 - alpha
            self._code_cooccurrence += cooccur_mask * alpha / bsz

            # I can't work out how to do this natively :(
            entropy_losses = []
            for bix in range(bsz):
                this_weight = 1
                for hix in range(len(vq_codes)):
                    this_weight *= all_encodings[bix, hix, cooccur_ixs[bix][hix].item()]

                # print(bix, this_weight, self._code_cooccurrence[cooccur_ixs[bix]])

                this_loss = torch.log(1 + 1e-10 - self._code_cooccurrence[cooccur_ixs[bix]]) * this_weight * -1
                entropy_losses.append(this_loss.unsqueeze(0))
                # print(this_loss)
            # print(entropy_losses)
            # exit()

            # if global_step > 200:

            #     print(cooccur_mask)
            #     print(self._code_cooccurrence)
            #     print(entropy_losses)
            #     exit()
            code_entropy_loss = torch.cat(entropy_losses, dim=0)

        # Loss
        if not self._ema:
            q_latent_loss = (
                nn.functional.mse_loss(quantized, inputs.detach(), reduction="none").mean(dim=-1).mean(dim=-1)
            )
        else:
            q_latent_loss = 0
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs, reduction="none").mean(dim=-1).mean(dim=-1)

        loss = self._commitment_cost * e_latent_loss + q_latent_loss + self._code_entropy_weight * code_entropy_loss

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
            if self._warmup_steps is not None and global_step is not None:
                w = min(global_step / self._warmup_steps, 1.0)
                quantized = inputs + (w * quantized - w * inputs).detach()
            else:
                quantized = inputs + (quantized - inputs).detach()

        return loss, quantized, vq_codes
