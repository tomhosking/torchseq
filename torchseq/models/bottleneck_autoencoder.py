import torch
import torch.nn as nn

from torchseq.models.encoder import SequenceEncoder
from torchseq.models.decoder import SequenceDecoder
from torchseq.models.bottleneck import PoolingBottleneck
from torchseq.models.modular_bottleneck import ModularBottleneck
from torchseq.utils.logging import Logger
from torchseq.utils.functions import cos_sim, onehot
from torchseq.models.lr_schedule import get_hyperbolic_schedule, get_tanh_schedule
from torchseq.models.vq_code_predictor import VQCodePredictor
from torchseq.models.pooling import MultiHeadedPooling


class BottleneckAutoencoderModel(nn.Module):
    def __init__(self, config, src_field="s1"):
        super().__init__()
        self.config = config

        # TEMP: deprecation warning
        if self.config.bottleneck.get("num_similar_heads", None) is not None:
            print('num_similar_heads is deprecated! Use "splice_head_offset" instead')

        num_heads = self.config.bottleneck.get("num_heads", self.config.encdec.get("num_heads", 1))

        if self.config.bottleneck.get("num_similar_heads", None) is not None:
            splice_head_offset = self.config.bottleneck.num_similar_heads
        else:
            splice_head_offset = self.config.bottleneck.get("splice_head_offset", num_heads)

        self.sep_splice_ix = self.config.bottleneck.embedding_dim // num_heads * splice_head_offset

        self.src_field = src_field

        self.seq_encoder = SequenceEncoder(config)

        if config.bottleneck.get("modular", False):
            self.bottleneck = ModularBottleneck(config)
        else:
            self.bottleneck = PoolingBottleneck(config)

        self.seq_decoder = SequenceDecoder(config, embeddings=self.seq_encoder.embeddings)

        if self.config.bottleneck.get("split_encoder", False):
            self.seq_encoder_2 = SequenceEncoder(config)
            self.bottleneck_2 = PoolingBottleneck(config)

            self.split_projection_1 = nn.utils.weight_norm(
                nn.Linear(config.encoder.embedding_dim, config.decoder.embedding_dim // 2, bias=False)
            )
            self.split_projection_2 = nn.utils.weight_norm(
                nn.Linear(config.encoder.embedding_dim, config.decoder.embedding_dim // 2, bias=False)
            )

        if self.config.bottleneck.get("code_predictor", None) is not None:
            pred_config = self.config.bottleneck.code_predictor
            if self.config.bottleneck.get("quantizer_transitions", False):
                vq_transitions = self.bottleneck.quantizer._transitions
            else:
                vq_transitions = None

            if self.config.bottleneck.get("modular", False):
                # HACK!
                vq_embeddings = self.bottleneck.module_list[1].quantizer._embedding
            else:
                vq_embeddings = self.bottleneck.quantizer._embedding

            self.code_predictor = VQCodePredictor(pred_config, transitions=vq_transitions, embeddings=vq_embeddings)

        if self.config.bottleneck.get("reduce_fn", "max") == "pool":
            if self.config.bottleneck.get(
                "code_predictor", None
            ) is not None and not self.config.bottleneck.code_predictor.get("sem_only", False):
                raise Exception("If pooling is used as the reduce_fn, sem_only must be set to true!")

            pool_dim = self.config.bottleneck.get("reduce_fn_dim", self.sep_splice_ix)
            if pool_dim != self.sep_splice_ix:
                self.reduce_proj = nn.Linear(self.sep_splice_ix, pool_dim, bias=False)
            else:
                self.reduce_proj = None

            self.reduce_pooling = MultiHeadedPooling(
                self.config.bottleneck.get("reduce_fn_heads", 1),
                pool_dim,
                dropout=self.config.dropout,
                use_final_linear=False,
            )

    def reduce_fn(self, inputs):
        if self.config.bottleneck.get("reduce_fn", "max") == "pool":
            if self.reduce_proj is not None:
                inputs = self.reduce_proj(inputs)
            reduced = self.reduce_pooling(inputs, inputs).unsqueeze(1)
        else:
            reduced = inputs.max(dim=1).values.unsqueeze(1)
        return reduced

    def forward(self, batch, output, memory=None, tgt_field=None):
        if memory is None:
            memory = {}

        # First pass? Construct the encoding
        if "encoding" not in memory:
            encoding, memory = self.seq_encoder(
                batch[self.src_field],
                batch[self.src_field + "_len"],
                memory,
                include_position=self.config.encoder.get("position_embeddings", True),
            )

            encoding_pooled, memory = self.bottleneck(
                encoding, memory, batch["_global_step"], head_mask=batch.get("head_mask", None)
            )

            prebn_encoding_pooled = memory["encoding_pooled"]

            if self.config.bottleneck.get(
                "code_predictor", None
            ) is not None and self.config.bottleneck.code_predictor.get("infer_codes", False):
                if self.config.bottleneck.code_predictor.get("sem_only", False):
                    codepred_input = (
                        encoding_pooled[:, :, : self.sep_splice_ix]
                        if self.config.bottleneck.code_predictor.get("post_bottleneck", False)
                        else prebn_encoding_pooled[:, :, : self.sep_splice_ix]
                    )
                else:
                    codepred_input = (
                        encoding_pooled
                        if self.config.bottleneck.code_predictor.get("post_bottleneck", False)
                        else prebn_encoding_pooled
                    )

                pred_codes = self.code_predictor.infer(
                    self.reduce_fn(codepred_input).squeeze(1), batch, outputs_to_block=memory.get("vq_codes")
                )
                batch["forced_codes"] = pred_codes

            if self.config.bottleneck.get("split_encoder", False):
                encoding2, memory = self.seq_encoder(batch[self.src_field], batch[self.src_field + "_len"], memory)
                encoding_pooled2, memory = self.bottleneck(
                    encoding2,
                    memory,
                    batch["_global_step"],
                    forced_codes=batch.get("forced_codes", None),
                    head_mask=batch.get("head_mask", None),
                )

                # TODO: Instead of 2x full size encoders + down projection, change to 2x half size encoders
                if self.config.encoder.embedding_dim != self.config.decoder.embedding_dim:
                    encoding_pooled = torch.cat(
                        [self.split_projection_1(encoding_pooled), self.split_projection_2(encoding_pooled2)], -1
                    )
                else:
                    encoding_pooled = torch.cat([encoding_pooled, encoding_pooled2], -1)

            if "template" in batch:
                template_memory = {}
                template_encoding, template_memory = self.seq_encoder(
                    batch["template"],
                    batch["template_len"],
                    template_memory,
                    include_position=self.config.encoder.get("template_position_embeddings", True),
                )

                if "forced_templ_encoding" in batch:
                    # print(batch["forced_templ_encoding"].shape, template_encoding)
                    template_encoding_pooled = batch["forced_templ_encoding"].unsqueeze(1)
                    template_memory["encoding_pooled"] = batch["forced_templ_encoding"].unsqueeze(1)
                else:
                    template_encoding_pooled, template_memory = self.bottleneck(
                        template_encoding,
                        template_memory,
                        batch["_global_step"],
                        forced_codes=batch.get("forced_codes", None),
                        head_mask=batch.get("head_mask", None),
                    )

                if "loss" in memory:
                    memory["loss"] += template_memory["loss"]

                if self.config.bottleneck.get("split_encoder", False):
                    template_encoding2, memory = self.seq_encoder(
                        batch["template"], batch["template_len"], template_memory
                    )
                    template_encoding_pooled2, memory = self.bottleneck(
                        template_encoding2, template_memory, batch["_global_step"]
                    )

                    template_encoding_pooled_joint = torch.cat(
                        [
                            self.split_projection_1(template_encoding_pooled),
                            self.split_projection_2(template_encoding_pooled2),
                        ],
                        -1,
                    )

                    template_encoding_pooled = template_encoding_pooled_joint

                if self.config.bottleneck.get("use_templ_encoding", False):
                    if "vq_codes" in template_memory:
                        memory["vq_codes_sem"] = memory["vq_codes"]
                        memory["vq_codes"] = template_memory["vq_codes"]

                    if self.config.bottleneck.get("invert_templ", False):
                        raise Exception("invert_templ is no longer supported")

                    sem_encoding_pooled = encoding_pooled
                    prebn_sem_encoding_pooled = prebn_encoding_pooled

                    encoding_pooled = torch.cat(
                        [
                            encoding_pooled[:, :, : self.sep_splice_ix],
                            template_encoding_pooled[:, :1, self.sep_splice_ix :].expand(
                                -1, encoding_pooled.shape[1], -1
                            ),
                        ],
                        dim=2,
                    )
                    templ_prebn_enc = template_memory["encoding_pooled"][:, :1, self.sep_splice_ix :].expand(
                        -1, prebn_encoding_pooled.shape[1], -1
                    )
                    prebn_encoding_pooled = torch.cat(
                        [prebn_encoding_pooled[:, :, : self.sep_splice_ix], templ_prebn_enc], dim=2
                    )

                if self.config.bottleneck.get("joint_train_codepred", False):
                    if "loss" not in memory:
                        memory["loss"] = 0

                    if self.config.bottleneck.code_predictor.get("sem_only", False):
                        codepred_input = self.reduce_fn(
                            sem_encoding_pooled[:, :, : self.sep_splice_ix]
                            if self.config.bottleneck.code_predictor.get("post_bottleneck", False)
                            else prebn_sem_encoding_pooled[:, :, : self.sep_splice_ix]
                        )
                    else:
                        codepred_input = self.reduce_fn(
                            sem_encoding_pooled
                            if self.config.bottleneck.code_predictor.get("post_bottleneck", False)
                            else prebn_sem_encoding_pooled
                        )

                    codepred_tgt = onehot(template_memory["vq_codes"], N=self.code_predictor.config.output_dim)

                    if self.config.bottleneck.code_predictor.get("force_unique", False):
                        codepred_tgt = codepred_tgt - onehot(
                            memory["vq_codes_sem"], N=self.code_predictor.config.output_dim
                        )
                        codepred_tgt = torch.max(codepred_tgt, torch.zeros_like(codepred_tgt))

                    codepred_loss = self.code_predictor.train_step(
                        codepred_input.squeeze(1),
                        codepred_tgt.detach(),
                        take_step=False,
                    )

                    if batch["_global_step"] % self.config.training.log_interval == 0 or not self.training:
                        Logger().log_scalar(
                            "bottleneck/codepred_loss" + ("_dev" if not self.training else ""),
                            codepred_loss.mean(),
                            batch["_global_step"],
                        )

                    loss_scale = 1.0

                    gamma = self.config.bottleneck.get("joint_train_codepred_schedule_gamma", None)
                    if gamma is not None:
                        loss_scale *= get_hyperbolic_schedule(gamma, batch["_global_step"])
                        codepred_loss = codepred_loss * loss_scale

                    tanh_gamma = self.config.bottleneck.get("joint_train_codepred_schedule_tanh_scale", None)
                    if tanh_gamma is not None:
                        loss_scale *= get_tanh_schedule(tanh_gamma, batch["_global_step"])

                    if batch["_global_step"] % self.config.training.log_interval == 0:
                        Logger().log_scalar("bottleneck/codepred_loss_scale", loss_scale, batch["_global_step"])

                    if batch["_global_step"] > self.config.bottleneck.get("joint_train_codepred_warmup_steps", 0):
                        memory["loss"] += codepred_loss * (
                            loss_scale * self.config.bottleneck.get("joint_train_codepred_weight", 1.0)
                            if self.training
                            else 1.0
                        )

                if self.config.bottleneck.get("separation_loss_weight", 0) > 0:
                    if "loss" not in memory:
                        memory["loss"] = 0

                    # TODO: there must be a cleaner way of flipping the tensor ranges here...
                    if self.config.bottleneck.get("cos_separation_loss", True):
                        diff1 = cos_sim(
                            encoding_pooled[:, :, self.sep_splice_ix :],
                            template_encoding_pooled[:, :, self.sep_splice_ix :],
                        )
                        diff2 = cos_sim(
                            encoding_pooled[:, :, : self.sep_splice_ix],
                            template_encoding_pooled[:, :, : self.sep_splice_ix],
                        )
                        if self.config.bottleneck.get("flip_separation_loss", False):
                            similarity_loss = 1 - 1 * diff1.mean(dim=-1).mean(dim=-1)
                            dissimilarity_loss = 1 + 1 * diff2.mean(dim=-1).mean(dim=-1)
                        else:
                            similarity_loss = 1 - 1 * diff2.mean(dim=-1).mean(dim=-1)
                            dissimilarity_loss = 1 + 1 * diff1.mean(dim=-1).mean(dim=-1)
                    else:
                        if self.config.bottleneck.get("flip_separation_loss", False):
                            diff1 = (
                                encoding_pooled[:, :, self.sep_splice_ix :]
                                - template_encoding_pooled[:, :, self.sep_splice_ix :]
                            )
                            similarity_loss = (diff1 ** 2).mean(dim=-1).mean(dim=-1)

                            diff2 = (
                                encoding_pooled[:, :, : self.sep_splice_ix]
                                - template_encoding_pooled[:, :, : self.sep_splice_ix]
                            )
                            dissimilarity_loss = torch.log(1 + 1 / (1e-2 + (diff2) ** 2)).mean(dim=-1).mean(dim=-1)
                        else:
                            diff1 = (
                                encoding_pooled[:, :, : self.sep_splice_ix]
                                - template_encoding_pooled[:, :, : self.sep_splice_ix]
                            )
                            similarity_loss = (diff1 ** 2).mean(dim=-1).mean(dim=-1)

                            diff2 = (
                                encoding_pooled[:, :, self.sep_splice_ix :]
                                - template_encoding_pooled[:, :, self.sep_splice_ix :]
                            )
                            dissimilarity_loss = torch.log(1 + 1 / (1e-2 + (diff2) ** 2)).mean(dim=-1).mean(dim=-1)

                        similarity_loss *= 10

                    separation_loss = (similarity_loss + dissimilarity_loss) * self.config.bottleneck.get(
                        "separation_loss_weight", 0
                    )

                    if self.training:
                        Logger().log_scalar("bottleneck/sim_loss", similarity_loss.mean(), batch["_global_step"])
                        Logger().log_scalar("bottleneck/dissim_loss", dissimilarity_loss.mean(), batch["_global_step"])

                    memory["loss"] += separation_loss

            memory["encoding"] = encoding_pooled
            memory["encoding_mask"] = None

            # Note: this returns the encodings *before* bottleneck!
            if self.sep_splice_ix > 0:
                memory["sep_encoding_1"] = self.reduce_fn(prebn_encoding_pooled[:, :, : self.sep_splice_ix]).detach()
                memory["sep_encoding_1_after_bottleneck"] = self.reduce_fn(
                    encoding_pooled[:, :, : self.sep_splice_ix]
                ).detach()
            memory["sep_encoding_2"] = (
                prebn_encoding_pooled[:, :, self.sep_splice_ix :].max(dim=1, keepdim=True).values.detach()
            )
            memory["sep_encoding_2_after_bottleneck"] = (
                encoding_pooled[:, :, self.sep_splice_ix :].max(dim=1, keepdim=True).values.detach()
            )

        # Fwd pass through decoder block
        logits, memory = self.seq_decoder(output, memory)

        return logits, memory
