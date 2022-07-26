from math import sqrt, exp
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Code inspired from https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py"""

from pydantic.dataclasses import dataclass
from torchseq.utils.logging import Logger


@dataclass
class VQVAEConfig:
    r"""
    Vector Quentized VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        commitment_loss_factor (float): The commitment loss factor in the loss. Default: 0.25.
        quantization_loss_factor: The quantization loss factor in the loss. Default: 1.
        num_embedding (int): The number of embedding points. Default: 512
        use_ema (bool): Whether to use the Exponential Movng Average Update (EMA). Default: False.
        decay (float): The decay to apply in the EMA update. Must be in [0, 1]. Default: 0.99.
    """
    latent_dim: int = 10
    commitment_loss_factor: float = 0.25
    quantization_loss_factor: float = 1.0
    num_embeddings: int = 512
    use_ema: bool = False
    decay: float = 0.99
    use_gumbel: bool = False
    gumbel_temp: float = 1.0
    gumbel_hard: bool = True
    temp_schedule: bool = False
    temp_init: float = 1.0
    temp_min: float = 0.5
    kl_loss_factor: float = 0.0
    normalize_inputs: bool = False
    batchnorm_inputs: bool = False
    normalize_embeds: float = None
    demean_inputs: bool = False
    noise_inputs: bool = False
    noise_outputs: bool = False
    hierarchy_depth: int = 1
    depth_dropout: float = 0.1
    depth_decay_factor: float = 0.5
    kl_warmup_steps: int = 0

    def __post_init_post_parse__(self):
        if self.use_ema:
            assert 0 <= self.decay <= 1, "The decay in the EMA update must be in [0, 1]. " f"Got {self.decay}."


class Quantizer(nn.Module):
    def __init__(self, model_config: VQVAEConfig):

        nn.Module.__init__(self)

        self.model_config = model_config

        self.embedding_dim = model_config.latent_dim
        self.num_embeddings = model_config.num_embeddings
        self.commitment_loss_factor = model_config.commitment_loss_factor
        self.quantization_loss_factor = model_config.quantization_loss_factor
        self.use_gumbel = model_config.use_gumbel
        self.use_ema = model_config.use_ema
        self.decay = model_config.decay
        self.demean_inputs = model_config.demean_inputs

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.embeddings.weight.data.uniform_(-1 / sqrt(self.embedding_dim), 1 / sqrt(self.embedding_dim))

        if model_config.normalize_embeds is not None:
            self.embeddings.weight.data = (
                self.embeddings.weight.data
                / torch.linalg.vector_norm(self.embeddings.weight, dim=-1, keepdim=True)
                * model_config.normalize_embeds
            )

        if model_config.batchnorm_inputs:
            self.batchnorm = nn.BatchNorm1d(self.embedding_dim, affine=False, track_running_stats=True)

        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))

        self.ema_embed = nn.Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))

        self.ema_embed.data.uniform_(-1 / sqrt(self.embedding_dim), 1 / sqrt(self.embedding_dim))

        # self.batch_norm = nn.BatchNorm1d(self.embedding_dim, affine=False)
        if self.demean_inputs:
            self.register_buffer("input_mean", torch.zeros(self.embedding_dim))

    def forward(self, z: torch.Tensor, step: int = 0):

        if self.model_config.noise_inputs and self.training:
            noise = torch.empty_like(z).normal_() * 0.1

            z = z + noise

        if self.model_config.batchnorm_inputs:
            z_shape = z.shape
            z = self.batchnorm(z.reshape(-1, self.embedding_dim)).reshape(z_shape) / sqrt(self.embedding_dim)

        if self.model_config.normalize_inputs:
            z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True) / sqrt(self.embedding_dim)

        if self.demean_inputs:
            # update mean
            if self.training:
                alpha = 0.99
                self.input_mean = alpha * self.input_mean + (1 - alpha) * z.mean(dim=0).squeeze().detach()

            # then subtract
            z = z - self.input_mean.detach()

        # print(z.shape)
        # print(torch.linalg.norm(z, dim=-1))
        # print(torch.linalg.norm(self.embeddings.weight, dim=1).min())
        # print(torch.linalg.norm(self.embeddings.weight, dim=1).max())
        # print(torch.linalg.norm(self.embeddings.weight, dim=1).mean())
        # print(torch.linalg.norm(self.embeddings.weight, dim=1))

        # raise Exception('quit!')

        distances = (
            (z.reshape(-1, self.embedding_dim) ** 2).sum(dim=-1, keepdim=True)
            + (self.embeddings.weight**2).sum(dim=-1)
            - 2 * z.reshape(-1, self.embedding_dim) @ self.embeddings.weight.T
        )

        closest = distances.argmin(-1).unsqueeze(-1)

        quantized_indices = closest.reshape(*z.shape[:-1])

        if self.model_config.temp_schedule:
            temp = max(self.model_config.temp_init * exp(-0.00003 * step), self.model_config.temp_min)
        else:
            temp = self.model_config.gumbel_temp

        if self.use_gumbel and self.training:
            one_hot_encoding = F.gumbel_softmax(
                -1.0 * distances, tau=temp, hard=self.model_config.gumbel_hard, dim=-1
            ).squeeze(1)
        else:
            one_hot_encoding = F.one_hot(closest, num_classes=self.num_embeddings).type(torch.float).squeeze(1)

        # quantization
        quantized = one_hot_encoding @ self.embeddings.weight
        quantized = quantized.reshape_as(z)

        if self.use_ema and self.training:

            n_i = torch.sum(one_hot_encoding, dim=0)

            self.cluster_size = self.cluster_size * self.decay + n_i * (1 - self.decay)

            dw = one_hot_encoding.T @ z.reshape(-1, self.embedding_dim)

            self.ema_embed.data = self.ema_embed * self.decay + dw * (1 - self.decay)

            n = torch.sum(self.cluster_size)

            self.cluster_size = ((self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n).detach()

            self.embeddings.weight.data = self.ema_embed / self.cluster_size.unsqueeze(-1)

        commitment_loss = F.mse_loss(
            quantized.detach().reshape(-1, self.embedding_dim),
            z.reshape(-1, self.embedding_dim),
            reduction="mean",
        )

        embedding_loss = F.mse_loss(
            quantized.reshape(-1, self.embedding_dim),
            z.detach().reshape(-1, self.embedding_dim),
            reduction="mean",
        ).mean(dim=-1)

        kl_loss = F.kl_div(
            F.log_softmax(-1.0 * distances, dim=-1),
            torch.ones_like(distances) / torch.ones_like(distances).sum(dim=-1, keepdim=True),
            reduction="batchmean",
        )
        kl_warmup_weight = (
            min(float(step) / float(self.model_config.kl_warmup_steps), 1.0)
            if self.model_config.kl_warmup_steps > 0
            else 1.0
        )

        if not self.use_gumbel:
            # straight through estimator
            quantized = z + (quantized - z).detach()
        else:
            quantized = quantized  # + z - z.detach()

        loss = (
            commitment_loss * self.commitment_loss_factor
            + embedding_loss * self.quantization_loss_factor
            + kl_loss * self.model_config.kl_loss_factor * kl_warmup_weight
        )

        if self.model_config.noise_outputs:
            noise = torch.empty_like(quantized).normal_() * 0.1

            quantized = quantized + noise

        # print(quantized.shape)
        # print(quantized_indices.shape)
        # quantized = quantized.permute(0, 3, 1, 2)

        # output = ModelOutput(
        #     quantized_vector=quantized,
        #     quantized_indices=quantized_indices.unsqueeze(1),
        #     loss=loss,
        # )
        global_step = step
        dev_str = "train" if self.training else "dev"
        head_ix = 0

        Logger().log_scalar(f"hrq_{dev_str}/{head_ix}/kl", kl_loss.mean(), global_step)

        Logger().log_scalar(f"hrq_{dev_str}/{head_ix}/temp", temp, global_step)

        posterior = F.softmax(-1.0 * distances, dim=-1)

        Logger().log_scalar(
            f"hrq_{dev_str}/{head_ix}/probs_ent",
            -1.0 * torch.sum(posterior * torch.log(posterior + 1e-10), dim=1).mean(),
            global_step,
        )
        Logger().log_scalar(
            f"hrq_{dev_str}/{head_ix}/probs_ent_batch",
            -1.0
            * torch.sum(
                posterior / torch.sum(posterior, dim=0) * torch.log(posterior / torch.sum(posterior, dim=0) + 1e-10),
                dim=0,
            ).mean(),
            global_step,
        )
        Logger().log_scalar(f"hrq_{dev_str}/{head_ix}/probs_min", torch.min(posterior), global_step)
        Logger().log_scalar(f"hrq_{dev_str}/{head_ix}/probs_max", torch.max(posterior), global_step)
        # Logger().log_histogram(f"hrq_{dev_str}/probs_hist_" + str(head_ix), probs.cpu().detach().tolist(), global_step)
        Logger().log_scalar(
            f"hrq_{dev_str}/{head_ix}/norm_input", torch.linalg.vector_norm(z, dim=-1).mean(), global_step
        )
        Logger().log_scalar(
            f"hrq_{dev_str}/{head_ix}/norm_meaninput",
            torch.linalg.vector_norm(z.mean(dim=0)),
            global_step,
        )
        Logger().log_scalar(
            f"hrq_{dev_str}/{head_ix}/norm_embed",
            torch.linalg.vector_norm(self.embeddings.weight, dim=1).mean(),
            global_step,
        )
        Logger().log_scalar(f"hrq_{dev_str}/commitment_loss", commitment_loss.mean(), global_step)

        return loss, quantized, quantized_indices.unbind(dim=1)


class PythaeQuantizerWrapper(Quantizer):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        num_heads=3,
        code_offset=0,
        warmup_steps=None,
        use_cosine_similarities=False,
        gumbel_temp=2.0,
        temp_min=0.5,
        temp_schedule=True,
        temp_schedule_gamma=10000,
        norm_loss_weight=None,
        init_decay_weight=0.5,
        init_embeds_xavier=True,
        init_embeds_truncnorm=False,
        init_embeds_uniform=False,
        init_embeds_polar=False,
        init_delay_steps=None,
        init_dynamic_var=False,
        init_scale=1.0,
        head_dropout=0.3,
        head_dropout_keep_first=False,
        learnable_priors=False,
        include_residual=False,
        residual_penalty=0.0,
        adaptive_depth=False,
        adaptive_penalty_weight=0.0,
        residual_warmup_steps=0,
        residual_dropout_steps=0,
        kmeans_delay=0,
        kl_weight=0.0,
        pre_norm=False,
        post_linear=False,
        init_sphere=False,
        soft_gumbel=False,
        output_seq=False,
        output_cumsum=False,
        simple_norm=False,
        logits_warmup_steps=0,
        sqrt_distances=False,
        learned_cluster_variances=False,
        kl_warmup_steps=0,
        commitment_weight=0,
        freeze_embeddings=False,
        detach=False,
        demean_inputs=False,
        noise_inputs=False,
    ):

        config = VQVAEConfig(
            latent_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_loss_factor=commitment_weight,
            quantization_loss_factor=0.0,
            use_ema=False,
            decay=0.99,
            use_gumbel=True,
            gumbel_temp=gumbel_temp,
            gumbel_hard=not soft_gumbel,
            temp_schedule=temp_schedule,
            temp_init=gumbel_temp,
            temp_min=temp_min,
            kl_loss_factor=kl_weight,
            normalize_inputs=simple_norm,
            batchnorm_inputs=pre_norm,
            normalize_embeds=(init_scale if init_sphere else None),
            demean_inputs=demean_inputs,
            noise_inputs=noise_inputs,
            noise_outputs=False,
            hierarchy_depth=num_heads,
            depth_dropout=head_dropout,
            depth_decay_factor=init_decay_weight,
            kl_warmup_steps=kl_warmup_steps,
        )
        super(PythaeQuantizerWrapper, self).__init__(config)

    def forward(self, inputs, global_step=None, forced_codes=None, head_mask=None, residual_mask=None):
        return super(PythaeQuantizerWrapper, self).forward(inputs, step=global_step)
