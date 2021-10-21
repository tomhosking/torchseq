import torch
import torch.nn as nn
from transformers import BartModel, BertModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.vq_vae import VectorQuantizerMultiHead
from torchseq.models.kl_divergence import gaussian_kl
from torchseq.models.vmf import vMF
from torchseq.utils.functions import reparameterize_gaussian

import torch.autograd.profiler as profiler


class ModularBottleneck(nn.Module):
    def __init__(self, config, embeddings=None):
        super().__init__()
        self.config = config

        self.ranges = []
        modules = []
        for module_config in config.bottleneck.modules:
            module_range = module_config.range
            module_dim = (
                self.config.bottleneck.embedding_dim
                // self.config.bottleneck.num_heads
                * (module_range[1] - module_range[0])
            )
            module = BottleneckPart(module_config, config, module_dim, module_range[1] - module_range[0])

            modules.append(module)
            self.ranges.append(module_range)

        self.module_list = nn.ModuleList(modules)

    def forward(self, encoding, memory, global_step, forced_codes=None, head_mask=None):

        # if head_mask is not None:
        #     print('hm in bottleneck')

        encodings_post = []
        encodings_pooled = []

        all_pooled = True
        any_pooled = False

        for module_range, module in zip(self.ranges, self.module_list):
            start_ix = self.config.bottleneck.embedding_dim // self.config.bottleneck.num_heads * module_range[0]
            end_ix = self.config.bottleneck.embedding_dim // self.config.bottleneck.num_heads * module_range[1]
            sub_encoding_pre = encoding[:, :, start_ix:end_ix]
            all_pooled = all_pooled & module.config.get("pooling", False)
            any_pooled = any_pooled | module.config.get("pooling", False)

            sub_encoding_post, sub_encoding_pooled, memory = module(
                sub_encoding_pre, memory, global_step, forced_codes=forced_codes, head_mask=head_mask
            )
            encodings_post.append(sub_encoding_post)
            encodings_pooled.append(sub_encoding_pooled)

        if any_pooled and not all_pooled:
            # Before we concat, we have to expand the pooled heads along the seq dimension
            for hix, (module_range, module) in enumerate(zip(self.ranges, self.module_list)):
                start_ix = self.config.bottleneck.embedding_dim // self.config.bottleneck.num_heads * module_range[0]
                end_ix = self.config.bottleneck.embedding_dim // self.config.bottleneck.num_heads * module_range[1]

                encodings_post[hix] = encodings_post[hix].expand(-1, encoding.shape[1], -1)
                encodings_pooled[hix] = encodings_pooled[hix].expand(-1, encoding.shape[1], -1)

        memory["encoding_pooled"] = torch.cat(encodings_pooled, dim=-1).detach()
        full_encoding_post = torch.cat(encodings_post, dim=-1)

        return full_encoding_post, memory


class BottleneckPart(nn.Module):
    def __init__(self, config, global_config, embedding_dim, num_heads):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.embedding_dim = embedding_dim

        # print('building part, dim=',embedding_dim)

        if config.get("pooling", False):

            self.pooling = MultiHeadedPooling(
                num_heads,
                self.embedding_dim,
                dropout=global_config.dropout,
                use_final_linear=False,
                use_layer_norm=self.config.get("layer_norm", False),
            )
            # print('built pooling, dim=', self.embedding_dim)

        if config.get("type", None) == "vae":
            # Extra modules for a variational bottleneck
            self.logvar_pooling = MultiHeadedPooling(
                num_heads,
                self.embedding_dim,
                dropout=global_config.dropout,
                use_final_linear=False,
            )
            # print('built logvar pooling, dim=', self.embedding_dim)

        if config.get("type", None) == "vmf":
            raise Exception("VMF is not yet implemented for the modular bottleneck!")
            self.vmf = vMF(
                embedding_dim,
                embedding_dim,
                kappa=80,
            )

        if config.get("type", None) == "hyperbolic":
            raise Exception("Hyperbolic bottleneck is not yet implemented for the modular bottleneck!")

        # VQ-VAE bottleneck
        if config.get("type", None) == "vqvae":

            # num_quantizer_heads = config.get("quantizer_heads", 1)

            quantizer_kwargs = config.get("quantizer", {})
            quantizer_kwargs.pop("codebook_size")
            self.quantizer = VectorQuantizerMultiHead(
                config.quantizer.codebook_size,
                embedding_dim,
                # commitment_cost=0.25,
                # decay=0.99,
                # num_heads=self.config.get("quantizer_heads", 1),
                # residual=self.config.get("quantizer_residual", False),
                # code_offset=self.config.get("code_offset", 0),
                # soft_em=self.config.get("quantizer_soft", True),
                # ema=self.config.get("quantizer_ema", True),
                # use_gumbel=self.config.get("quantizer_gumbel", False),
                # gumbel_temp=self.config.get("quantizer_gumbel_temp", 1.0),
                # temp_schedule=self.config.get("quantizer_gumbel_temp_schedule", False),
                # use_straight_through=self.config.get("quantizer_straight_through", True),
                # warmup_steps=self.config.get("quantizer_warmup_steps", None),
                # code_entropy_weight=self.config.get("quantizer_entropy_weight", 0),
                # hierarchical=self.config.get("quantizer_hierarchical", False),
                # hierarchical_balance_dims=self.config.get("hierarchical_balance_dims", False),
                # transitions=self.config.get("quantizer_transitions", False),
                # transitions_bias=self.config.get("quantizer_transitions_bias", False),
                # transitions_embed=self.config.get("quantizer_transitions_embed", False),
                # transitions_log=self.config.get("quantizer_transitions_log", False),
                # relative_error=self.config.get("quantizer_relative_error", False),
                # use_cosine_similarities=self.config.get("quantizer_cosine", False),
                # separate_output_embedding=self.config.get("quantizer_separate_output_embedding", False),
                # use_code_classifier=self.config.get("quantizer_classifier", False),
                # additive=self.config.get("quantizer_additive", False),
                # only_final=self.config.get("quantizer_only_final", False),
                # norm_loss_weight=self.config.get("quantizer_norm_loss_weight", None),
                **quantizer_kwargs,
            )

    def forward(self, encoding, memory, global_step, forced_codes=None, head_mask=None):
        # if head_mask is not None:
        #     print('hm in bottleneck part')

        # print('BN part, input=', encoding.shape)

        encoding_pooled = (
            self.pooling(key=encoding, value=encoding).unsqueeze(1) if self.config.get("pooling", True) else encoding
        )

        # Stash this for later
        encoding_post = encoding_pooled

        # Quantize
        if self.config.get("type", None) == "vqvae":

            vq_loss, encoding_post, quantizer_indices = self.quantizer(
                encoding_post, global_step, forced_codes, head_mask
            )

            if "loss" not in memory:
                memory["loss"] = 0
            memory["loss"] += vq_loss
            memory["vq_codes"] = torch.cat([x.unsqueeze(1).detach() for x in quantizer_indices], dim=1)

            if forced_codes is not None:
                assert (
                    forced_codes.detach().tolist() == memory["vq_codes"].detach().tolist()
                ), "Forced codes != vq_codes assigned by quantizer!"

        if self.config.get("type", None) == "vmf":
            raise Exception("VMF is not yet implemented for the modular bottleneck!")

        var_weight = self.config.get("prior_var_weight", 1.0)

        if not isinstance(var_weight, float) and len(var_weight) > 1:
            raise Exception("Varying VAE noise weight is not yet supported for the modular bottleneck!")
            assert len(var_weight) == self.config.encdec.num_heads
            var_weight = torch.Tensor(var_weight).to(encoding.device)
            var_weight = torch.repeat_interleave(var_weight, self.config.embedding_dim // self.config.encdec.num_heads)

        # if self.config.get("hyperbolic", False):
        #     encoding_post, memory = self.hyperbolic_bottleneck(encoding_post, memory, global_step)

        # Reparameterise for VAE
        if self.config.get("variational", False) or self.config.get("type", None) == "vae":

            mu = encoding_post
            logvar = self.logvar_pooling(key=encoding, value=encoding).unsqueeze(1)

            encoding_post = reparameterize_gaussian(mu, logvar, var_weight=var_weight)

            kl_loss = torch.mean(gaussian_kl(mu, logvar), dim=1)

            kl_warmup_steps = self.global_config.training.data.get("kl_warmup_steps", 0)
            kl_weight_mult = self.global_config.training.data.get("kl_weight", 1.0)
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

        return encoding_post, encoding_pooled, memory
