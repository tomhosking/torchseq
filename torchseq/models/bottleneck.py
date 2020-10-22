import math

import torch
import torch.nn as nn
from transformers import BartModel, BertModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.positional_embeddings import PositionalEncoding
from torchseq.models.multihead_output import MultiHeadOutput
from torchseq.utils.tokenizer import Tokenizer
from torchseq.models.vq_vae import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerMultiHead
from torchseq.models.kl_divergence import get_kl


class PoolingBottleneck(nn.Module):
    def __init__(self, config, embeddings=None):
        super().__init__()
        self.config = config

        self.encoder_pooling = MultiHeadedPooling(
            config.encdec.num_heads,
            config.bottleneck.embedding_dim,
            dropout=config.dropout,
            model_dim_out=config.bottleneck.embedding_dim,
            use_final_linear=False,
        )

        # Extra modules for a variational bottleneck
        if self.config.bottleneck.get("variational", False):
            self.encoder_logvar_pooling = MultiHeadedPooling(
                config.encdec.num_heads,
                config.bottleneck.embedding_dim,
                dropout=config.dropout,
                model_dim_out=config.bottleneck.embedding_dim,
                use_final_linear=False,
            )

        # VQ-VAE bottleneck
        if self.config.bottleneck.get("vector_quantized", False):
            self.quantizer = VectorQuantizerMultiHead(
                self.config.bottleneck.codebook_size,
                self.config.bottleneck.embedding_dim,
                commitment_cost=0.25,
                decay=0.99,
                num_heads=self.config.bottleneck.get("quantizer_heads", 1),
                residual=self.config.bottleneck.get("quantizer_residual", False),
                code_offset=self.config.bottleneck.get("code_offset", 0),
                num_residual=self.config.bottleneck.get("quantizer_num_residual", 0),
                soft_em=self.config.bottleneck.get("quantizer_soft", True),
                warmup_steps=self.config.bottleneck.get("quantizer_warmup_steps", None),
            )

    def forward(self, encoding, memory, global_step):

        # Pool
        encoding_pooled = (
            self.encoder_pooling(key=encoding, value=encoding).unsqueeze(1)
            if self.config.bottleneck.get("pooling", True)
            else encoding
        )

        # Quantize
        if self.config.bottleneck.get("vector_quantized", False):
            vq_loss, encoding_pooled, quantizer_indices = self.quantizer(encoding_pooled, global_step)

            if "loss" not in memory:
                memory["loss"] = 0
            memory["loss"] += vq_loss
            memory["vq_codes"] = quantizer_indices

        # Reparameterise for VAE
        if self.config.bottleneck.get("variational", False):

            def reparameterize(mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)

                var_weight = self.config.bottleneck.get("prior_var_weight", 1.0)
                if not isinstance(var_weight, float) and len(var_weight) > 1:
                    assert len(var_weight) == self.config.encdec.num_heads
                    var_weight = torch.Tensor(var_weight).to(encoding.device)
                    var_weight = torch.repeat_interleave(
                        var_weight, self.config.bottleneck.embedding_dim // self.config.encdec.num_heads
                    )

                return mu + eps * std * var_weight

            # If VQ-VAE with residual path, only make the residual heads variational
            if (
                self.config.bottleneck.get("vector_quantized", False)
                and self.config.bottleneck.get("quantizer_num_residual", 0) > 0
            ):
                splice_ix = (
                    self.config.bottleneck.embedding_dim
                    // self.config.bottleneck.get("quantizer_heads", 1)
                    * self.config.bottleneck.get("quantizer_num_residual", 0)
                )

                mu = encoding_pooled
                logvar = self.encoder_logvar_pooling(key=encoding, value=encoding).unsqueeze(1)

                encoding_pooled[:, :1, :splice_ix] = reparameterize(mu[:, :1, :splice_ix], logvar[:, :1, :splice_ix])
            else:
                mu = encoding_pooled
                logvar = self.encoder_logvar_pooling(key=encoding, value=encoding).unsqueeze(1)

                encoding_pooled = reparameterize(mu, logvar)

            kl_loss = torch.mean(get_kl(mu, logvar), dim=1)

            kl_warmup_steps = self.config.training.data.get("kl_warmup_steps", 0)
            kl_weight = (
                1
                if global_step >= 2 * kl_warmup_steps
                else (
                    0
                    if global_step < kl_warmup_steps
                    else float(global_step - kl_warmup_steps) / (1.0 * kl_warmup_steps)
                )
            )

            if "loss" not in memory:
                memory["loss"] = 0
            memory["loss"] += kl_loss * kl_weight

        return encoding_pooled, memory
