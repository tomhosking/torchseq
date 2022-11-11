import torch
import torch.nn as nn
from transformers import BartModel, BertModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.vq_vae import VectorQuantizerMultiHead
from torchseq.models.hrq_vae import HierarchicalRefinementQuantizer
from torchseq.models.pythae_vq import PythaeQuantizerWrapper
from torchseq.models.kl_divergence import gaussian_kl
from torchseq.models.vmf import vMF
from torchseq.utils.functions import reparameterize_gaussian


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

        if config.bottleneck.get("input_dim", config.bottleneck.embedding_dim) != config.bottleneck.embedding_dim:
            self.input_projection = nn.Linear(config.bottleneck.input_dim, config.bottleneck.embedding_dim, bias=False)
        else:
            self.input_projection = None

        if config.bottleneck.get("output_dim", config.bottleneck.embedding_dim) != config.bottleneck.embedding_dim:
            self.output_projection = nn.Linear(
                config.bottleneck.embedding_dim, config.bottleneck.output_dim, bias=False
            )
        else:
            self.output_projection = None

        # Pre/post transformations
        if config.bottleneck.get("pre_transform", None) is not None:
            num_layers = config.bottleneck.pre_transform.get("num_layers", 0)
            hidden_dim = config.bottleneck.pre_transform.get("hidden_dim", config.bottleneck.embedding_dim * 4)
            num_heads = config.bottleneck.pre_transform.get("num_heads", 8)

            transform_layer = nn.TransformerEncoderLayer(
                config.bottleneck.get("input_dim", config.bottleneck.embedding_dim),
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=config.dropout,
                activation="relu",
                batch_first=True,
            )
            encoder_norm = nn.LayerNorm(config.bottleneck.get("input_dim", config.bottleneck.embedding_dim))
            self.pre_transform = nn.TransformerEncoder(
                transform_layer, num_layers, encoder_norm, enable_nested_tensor=True
            )
        else:
            self.pre_transform = None

        if config.bottleneck.get("post_transform", None) is not None:
            num_layers = config.bottleneck.post_transform.get("num_layers", 0)
            hidden_dim = config.bottleneck.post_transform.get("hidden_dim", config.bottleneck.embedding_dim * 4)
            num_heads = config.bottleneck.post_transform.get("num_heads", 8)

            transform_layer = nn.TransformerEncoderLayer(
                config.bottleneck.get("output_dim", config.bottleneck.embedding_dim),
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=config.dropout,
                activation="relu",
                batch_first=True,
            )
            transform_norm = nn.LayerNorm(config.bottleneck.get("output_dim", config.bottleneck.embedding_dim))
            self.post_transform = nn.TransformerEncoder(
                transform_layer, num_layers, transform_norm, enable_nested_tensor=True
            )
        else:
            self.post_transform = None

    def forward(self, encoding, memory, global_step, forced_codes=None, head_mask=None, residual_mask=None):

        # if head_mask is not None:
        #     print('hm in bottleneck')

        if self.pre_transform is not None:
            encoding = self.pre_transform(encoding)

        if self.input_projection is not None:
            encoding = self.input_projection(encoding)

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
                sub_encoding_pre,
                memory,
                global_step,
                forced_codes=forced_codes,
                head_mask=head_mask,
                residual_mask=residual_mask,
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

        if any_pooled:
            memory["encoding_mask"] = None

        if self.output_projection is not None:
            memory["encoding_pooled"] = self.output_projection(memory["encoding_pooled"])
            full_encoding_post = self.output_projection(full_encoding_post)

        if self.post_transform is not None:
            encoding = self.post_transform(encoding)

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

        if config.get("freeze_pooling", False):
            for p in self.pooling.parameters():
                p.requires_grad = False

        if self.config.get("variational", False) or self.config.get("type", None) == "vae":
            # Extra modules for a variational bottleneck
            self.logvar_pooling = MultiHeadedPooling(
                num_heads,
                self.embedding_dim,
                dropout=global_config.dropout,
                use_final_linear=False,
            )
            if not config.get("pooling", False):
                self.mu_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                self.var_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
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
                **quantizer_kwargs,
            )
        # HRQ-VAE bottleneck
        if config.get("type", None) == "hrqvae":
            quantizer_kwargs = config.get("quantizer", {})
            quantizer_kwargs.pop("codebook_size")
            self.quantizer = HierarchicalRefinementQuantizer(
                config.quantizer.codebook_size,
                embedding_dim,
                **quantizer_kwargs,
            )
        # VQ-VAE bottleneck (pythae implementation)
        if config.get("type", None) == "pythae:vqvae":
            quantizer_kwargs = config.get("quantizer", {})
            quantizer_kwargs.pop("codebook_size")
            self.quantizer = PythaeQuantizerWrapper(
                config.quantizer.codebook_size,
                embedding_dim,
                **quantizer_kwargs,
            )

    def forward(self, encoding, memory, global_step, forced_codes=None, head_mask=None, residual_mask=None):
        # if head_mask is not None:
        #     print('hm in bottleneck part')

        # print('BN part, input=', encoding.shape)

        encoding_pooled = (
            self.pooling(key=encoding, value=encoding, mask=memory["encoding_mask"]).unsqueeze(1)
            if self.config.get("pooling", True)
            else encoding
        )

        # Stash this for later
        encoding_post = encoding_pooled

        # Quantize
        if self.config.get("type", None) in ["vqvae", "hrqvae", "pythae:vqvae"]:

            if self.config.get("type", None) == "hrqvae":
                vq_loss, encoding_post, quantizer_indices = self.quantizer(
                    encoding_post, global_step, forced_codes, head_mask, residual_mask
                )
            else:
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

        var_weight = self.config.get("prior_var_weight", 1.0) * (
            1.0 if self.training or not self.global_config.eval.get("vae_use_map", True) else 0.0
        )

        if not isinstance(var_weight, float) and len(var_weight) > 1:
            raise Exception("Varying VAE noise weight is not yet supported for the modular bottleneck!")
            assert len(var_weight) == self.config.encoder.num_heads
            var_weight = torch.Tensor(var_weight).to(encoding.device)
            var_weight = torch.repeat_interleave(
                var_weight, self.config.embedding_dim // self.config.encoder.num_heads
            )

        # if self.config.get("hyperbolic", False):
        #     encoding_post, memory = self.hyperbolic_bottleneck(encoding_post, memory, global_step)

        # Reparameterise for VAE
        if self.config.get("variational", False) or self.config.get("type", None) == "vae":

            if self.config.get("pooling", False):
                mu = encoding_post
                logvar = self.logvar_pooling(key=encoding, value=encoding, mask=memory["encoding_mask"]).unsqueeze(1)
            else:
                mu = self.mu_proj(encoding_post)
                logvar = self.var_proj(encoding_post)

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
