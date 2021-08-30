import torch
import torch.nn as nn
from transformers import BartModel, BertModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.vq_vae import VectorQuantizerMultiHead
from torchseq.models.kl_divergence import get_kl
from torchseq.models.vmf import vMF
from torchseq.utils.functions import reparameterize_gaussian

import torch.autograd.profiler as profiler


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

        if self.config.bottleneck.get("use_vmf", False):
            self.vmf = vMF(
                config.bottleneck.embedding_dim // self.config.bottleneck.get("quantizer_heads", 1),
                config.bottleneck.embedding_dim // self.config.bottleneck.get("quantizer_heads", 1),
                kappa=80,
            )

        # VQ-VAE bottleneck
        if self.config.bottleneck.get("vector_quantized", False):
            if self.config.bottleneck.get("residual_head_range", None) is None:
                self.residual_head_range = (0, self.config.bottleneck.get("quantizer_num_residual", 0))
            else:
                self.residual_head_range = self.config.bottleneck.get("residual_head_range", (0, 0))
                assert (
                    len(self.residual_head_range) == 2
                ), "bottlneck residual_head_range must be length 2! (lower, upper)"

            total_quantizer_heads = self.config.bottleneck.get("quantizer_heads", 1)
            num_quantizer_heads = total_quantizer_heads - self.residual_head_range[1] - self.residual_head_range[0]

            quantizer_kwargs = self.config.bottleneck.get("quantizer", {})
            self.quantizer = VectorQuantizerMultiHead(
                self.config.bottleneck.codebook_size,
                (self.config.bottleneck.embedding_dim * num_quantizer_heads) // total_quantizer_heads,
                commitment_cost=0.25,
                decay=0.99,
                num_heads=num_quantizer_heads,
                residual=self.config.bottleneck.get("quantizer_residual", False),
                code_offset=self.config.bottleneck.get("code_offset", 0),
                soft_em=self.config.bottleneck.get("quantizer_soft", True),
                ema=self.config.bottleneck.get("quantizer_ema", True),
                use_gumbel=self.config.bottleneck.get("quantizer_gumbel", False),
                gumbel_temp=self.config.bottleneck.get("quantizer_gumbel_temp", 1.0),
                temp_schedule=self.config.bottleneck.get("quantizer_gumbel_temp_schedule", False),
                use_straight_through=self.config.bottleneck.get("quantizer_straight_through", True),
                warmup_steps=self.config.bottleneck.get("quantizer_warmup_steps", None),
                code_entropy_weight=self.config.bottleneck.get("quantizer_entropy_weight", 0),
                hierarchical=self.config.bottleneck.get("quantizer_hierarchical", False),
                hierarchical_balance_dims=self.config.bottleneck.get("hierarchical_balance_dims", False),
                transitions=self.config.bottleneck.get("quantizer_transitions", False),
                transitions_bias=self.config.bottleneck.get("quantizer_transitions_bias", False),
                transitions_embed=self.config.bottleneck.get("quantizer_transitions_embed", False),
                transitions_log=self.config.bottleneck.get("quantizer_transitions_log", False),
                relative_error=self.config.bottleneck.get("quantizer_relative_error", False),
                use_cosine_similarities=self.config.bottleneck.get("quantizer_cosine", False),
                separate_output_embedding=self.config.bottleneck.get("quantizer_separate_output_embedding", False),
                use_code_classifier=self.config.bottleneck.get("quantizer_classifier", False),
                additive=self.config.bottleneck.get("quantizer_additive", False),
                only_final=self.config.bottleneck.get("quantizer_only_final", False),
                norm_loss_weight=self.config.bottleneck.get("quantizer_norm_loss_weight", None),
                **quantizer_kwargs,
            )

    def forward(self, encoding, memory, global_step, forced_codes=None):

        # Pool
        encoding_pooled = (
            self.encoder_pooling(key=encoding, value=encoding).unsqueeze(1)
            if self.config.bottleneck.get("pooling", True)
            else encoding
        )

        memory["encoding_pooled"] = encoding_pooled.detach()

        # Quantize
        if self.config.bottleneck.get("vector_quantized", False):
            # splice_begin, splice_end = self.residual_head_range
            # splice_begin *= self.config.bottleneck.embedding_dim // self.config.bottleneck.get("quantizer_heads", 1)
            # splice_end *= self.config.bottleneck.embedding_dim // self.config.bottleneck.get("quantizer_heads", 1)

            splice_ix = (
                self.config.bottleneck.embedding_dim
                // self.config.bottleneck.get("quantizer_heads", 1)
                * self.config.bottleneck.get("quantizer_num_residual", 0)
            )
            # if splice_begin > 0:
            #     raise Exception("Arbitrary quantizer residual ranges are not currently supported")

            vq_loss, quantized_encoding, quantizer_indices = self.quantizer(
                encoding_pooled[:, :, splice_ix:], global_step, forced_codes
            )

            encoding_pooled = torch.cat([encoding_pooled[:, :, :splice_ix], quantized_encoding], dim=-1)

            if "loss" not in memory:
                memory["loss"] = 0
            memory["loss"] += vq_loss
            memory["vq_codes"] = torch.cat([x.unsqueeze(1).detach() for x in quantizer_indices], dim=1)

            if forced_codes is not None:
                assert (
                    forced_codes.detach().tolist() == memory["vq_codes"].detach().tolist()
                ), "Forced codes != vq_codes assigned by quantizer!"

        if self.config.bottleneck.get("pooling_range", None) is not None:
            begin_hix, end_hix = self.config.bottleneck.get("pooling_range", (0, 0))
            begin_ix = (
                self.config.bottleneck.embedding_dim // self.config.bottleneck.get("quantizer_heads", 1) * begin_hix
            )
            end_ix = self.config.bottleneck.embedding_dim // self.config.bottleneck.get("quantizer_heads", 1) * end_hix
            encoding_pooled = torch.cat(
                [
                    encoding[:, :, :begin_ix],
                    encoding_pooled[:, :, begin_ix:end_ix].expand(-1, encoding.shape[1], -1),
                    encoding[:, :, end_ix:],
                ],
                dim=-1,
            )

        if self.config.bottleneck.get("use_vmf", False):
            if self.config.bottleneck.get("quantizer_heads", 1) > 1:
                num_heads = self.config.bottleneck.get("quantizer_heads", 1)
                encoding_chunked = torch.cat(torch.chunk(encoding_pooled, num_heads, -1), 1)

                splice_head_ix = self.config.bottleneck.get("quantizer_num_residual", 0)

                chunk_dim = encoding_chunked.size()[-1]
                encoding_chunked = encoding_chunked[:, splice_head_ix:, :].view(-1, chunk_dim)

            else:
                encoding_chunked = encoding_pooled.squeeze(1)

            tup, kld, encoding_vmf = self.vmf.build_bow_rep(encoding_chunked)

            if self.config.bottleneck.get("quantizer_heads", 1) > 1:
                bsz = encoding_pooled.size()[0]
                encoding_vmf = encoding_vmf.view(1, bsz, -1).transpose_(0, 1)

                if splice_head_ix > 0:
                    splice_ix = (
                        self.config.bottleneck.embedding_dim
                        // self.config.bottleneck.get("quantizer_heads", 1)
                        * self.config.bottleneck.get("quantizer_num_residual", 0)
                    )
                    encoding_pooled = torch.cat([encoding_vmf, encoding_pooled[:, :1, splice_ix:]], dim=-1)
                else:
                    encoding_pooled = encoding_vmf

                kld = kld.view(bsz, -1).sum(-1)

            if "loss" not in memory:
                memory["loss"] = 0
            memory["loss"] += kld

        var_weight = self.config.bottleneck.get("prior_var_weight", 1.0)

        if not isinstance(var_weight, float) and len(var_weight) > 1:
            assert len(var_weight) == self.config.encdec.num_heads
            var_weight = torch.Tensor(var_weight).to(encoding.device)
            var_weight = torch.repeat_interleave(
                var_weight, self.config.bottleneck.embedding_dim // self.config.encdec.num_heads
            )

        if self.config.bottleneck.get("hyperbolic", False):
            encoding_pooled, memory = self.hyperbolic_bottleneck(encoding_pooled, memory, global_step)

        # Reparameterise for VAE
        if self.config.bottleneck.get("variational", False):

            mu = encoding_pooled
            logvar = self.encoder_logvar_pooling(key=encoding, value=encoding).unsqueeze(1)

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

                if not isinstance(var_weight, float) and len(var_weight) > 1:
                    var_weight = var_weight[:splice_ix]

                # Reparametrisation trick, only for the residual heads
                var_encoding = reparameterize_gaussian(
                    mu[:, :, :splice_ix], logvar[:, :, :splice_ix], var_weight=var_weight
                )
                encoding_pooled = torch.cat([var_encoding, encoding_pooled[:, :, splice_ix:]], dim=-1)

                kl_loss = torch.mean(get_kl(mu[:, :, :splice_ix], logvar[:, :, :splice_ix]), dim=1)
            else:

                encoding_pooled = reparameterize_gaussian(mu, logvar, var_weight=var_weight)

                kl_loss = torch.mean(get_kl(mu, logvar), dim=1)

            kl_warmup_steps = self.config.training.data.get("kl_warmup_steps", 0)
            kl_weight_mult = self.config.training.data.get("kl_weight", 1.0)
            kl_weight = (
                1
                if global_step >= 2 * kl_warmup_steps
                else (
                    0
                    if global_step < kl_warmup_steps
                    else float(global_step - kl_warmup_steps) / (1.0 * kl_warmup_steps)
                )
            ) * kl_weight_mult

            if "loss" not in memory:
                memory["loss"] = 0
            memory["loss"] += kl_loss * kl_weight

        return encoding_pooled, memory
