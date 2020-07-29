import math

import torch
import torch.nn as nn
from transformers import BartModel, BertModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.positional_embeddings import PositionalEncoding
from torchseq.models.multihead_output import MultiHeadOutput
from torchseq.utils.tokenizer import Tokenizer
from torchseq.models.vq_vae import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerMultiHead


class PoolingBottleneck(nn.Module):
    def __init__(self, config, embeddings=None):
        super().__init__()
        self.config = config

        self.encoder_pooling = MultiHeadedPooling(
            config.encdec.num_heads,
            config.embedding_dim,
            dropout=config.dropout,
            model_dim_out=config.embedding_dim,
            use_final_linear=False,
        )

        # Extra modules for a variational bottleneck
        if self.config.encdec.data.get("variational", False):
            self.encoder_logvar_pooling = MultiHeadedPooling(
                config.encdec.num_heads,
                config.embedding_dim,
                dropout=config.dropout,
                model_dim_out=config.embedding_dim,
                use_final_linear=False,
            )

        # VQ-VAE bottleneck
        if self.config.encdec.data.get("vector_quantized", False):
            self.quantizer = VectorQuantizerMultiHead(
                self.config.encdec.codebook_size,
                self.config.embedding_dim,
                commitment_cost=0.25,
                decay=0.99,
                num_heads=self.config.encdec.get("quantizer_heads", 1),
                residual=self.config.encdec.get("quantizer_residual", False),
                code_offset=self.config.encdec.get("code_offset", 0),
                num_residual=self.config.encdec.get("quantizer_num_residual", 0),
            )

    def forward(self, encoding, memory):

        # Pool
        encoding_pooled = (
            self.encoder_pooling(key=encoding, value=encoding).unsqueeze(1)
            if self.config.encdec.data.get("pooling", True)
            else encoding
        )

        # Quantize
        if self.config.encdec.data.get("vector_quantized", False):
            vq_loss, encoding_pooled, quantizer_indices = self.quantizer(encoding_pooled)
            if "loss" not in memory:
                memory["loss"] = 0
            memory["loss"] += vq_loss
            memory["vq_codes"] = quantizer_indices

        # Reparameterise for VAE
        if self.config.encdec.data.get("variational", False):
            memory["mu"] = encoding_pooled
            memory["logvar"] = self.encoder_logvar_pooling(key=encoding, value=encoding).unsqueeze(1)

            def reparameterize(mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)

                var_weight = self.config.encdec.data.get("prior_var_weight", 1.0)
                if not isinstance(var_weight, float) and len(var_weight) > 1:
                    assert len(var_weight) == self.config.encdec.num_heads
                    var_weight = torch.Tensor(var_weight).to(encoding.device)
                    var_weight = torch.repeat_interleave(
                        var_weight, self.config.embedding_dim // self.config.encdec.num_heads
                    )

                return mu + eps * std * var_weight

            encoding_pooled = reparameterize(memory["mu"], memory["logvar"])

        return encoding_pooled, memory
