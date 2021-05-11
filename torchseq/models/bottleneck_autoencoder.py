import math

import torch
import torch.nn as nn

# from transformers import BartModel, BertModel

# from torchseq.models.pooling import MultiHeadedPooling
# from torchseq.models.positional_embeddings import PositionalEncoding
# from torchseq.models.multihead_output import MultiHeadOutput
# from torchseq.utils.tokenizer import Tokenizer
# from torchseq.models.vq_vae import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerMultiHead

from torchseq.models.encoder import SequenceEncoder
from torchseq.models.decoder import SequenceDecoder
from torchseq.models.bottleneck import PoolingBottleneck
from torchseq.utils.logging import Logger
from torchseq.utils.functions import cos_sim
from torchseq.models.vq_code_predictor import VQCodePredictor


class BottleneckAutoencoderModel(nn.Module):
    def __init__(self, config, src_field="s1"):
        super().__init__()
        self.config = config

        self.src_field = src_field

        self.seq_encoder = SequenceEncoder(config)
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
            print("Code predictor created")
            pred_config = self.config.bottleneck.code_predictor

            self.code_predictor = VQCodePredictor(pred_config)

    def forward(self, batch, output, memory=None, tgt_field=None):
        if memory is None:
            memory = {}

        # First pass? Construct the encoding
        if "encoding" not in memory:
            # print('main encoding: ', self.config.encoder.get("position_embeddings", True))
            # print('s1', batch["s1_text"])
            # print('s2', batch["s2_text"])
            encoding, memory = self.seq_encoder(
                batch[self.src_field],
                batch[self.src_field + "_len"],
                memory,
                include_position=self.config.encoder.get("position_embeddings", True),
            )

            encoding_pooled, memory = self.bottleneck(encoding, memory, batch["_global_step"])

            raw_encoding_pooled = memory["encoding_pooled"]

            if self.config.bottleneck.get(
                "code_predictor", None
            ) is not None and self.config.bottleneck.code_predictor.get("infer_codes", False):
                pred_codes = self.code_predictor.infer(raw_encoding_pooled)
                batch["forced_codes"] = pred_codes

            if self.config.bottleneck.get("split_encoder", False):
                encoding2, memory = self.seq_encoder(batch[self.src_field], batch[self.src_field + "_len"], memory)
                encoding_pooled2, memory = self.bottleneck(
                    encoding2, memory, batch["_global_step"], forced_codes=batch.get("forced_codes", None)
                )

                # TODO: Instead of 2x full size encoders + down projection, change to 2x half size encoders
                if self.config.encoder.embedding_dim != self.config.decoder.embedding_dim:
                    encoding_pooled = torch.cat(
                        [self.split_projection_1(encoding_pooled), self.split_projection_2(encoding_pooled2)], -1
                    )
                else:
                    encoding_pooled = torch.cat([encoding_pooled, encoding_pooled2], -1)

            splice_head_offset = (
                self.config.bottleneck.get("splice_head_offset", 0)
                if self.config.bottleneck.get("num_similar_heads", None) is None
                else self.config.bottleneck.get("num_similar_heads", 0)
            )
            sep_splice_ix = (
                self.config.decoder.embedding_dim // self.config.encdec.get("num_heads", 1) * splice_head_offset
            )

            if "template" in batch:
                template_memory = {}
                # print('templ encoding: ', self.config.encoder.get("template_position_embeddings", True))
                # print(batch["template_text"])
                template_encoding, template_memory = self.seq_encoder(
                    batch["template"],
                    batch["template_len"],
                    template_memory,
                    include_position=self.config.encoder.get("template_position_embeddings", True),
                )

                template_encoding_pooled, template_memory = self.bottleneck(
                    template_encoding,
                    template_memory,
                    batch["_global_step"],
                    forced_codes=batch.get("forced_codes", None),
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
                    if self.config.bottleneck.get("invert_templ", False):
                        raise Exception("invert_templ is no longer supported")
                        encoding_pooled[:, :, :sep_splice_ix] = template_encoding_pooled[:, :, :sep_splice_ix]
                        raw_encoding_pooled[:, :, :sep_splice_ix] = template_memory["encoding_pooled"][
                            :, :, :sep_splice_ix
                        ]
                    else:
                        encoding_pooled = torch.cat(
                            [encoding_pooled[:, :, :sep_splice_ix], template_encoding_pooled[:, :1, sep_splice_ix:]],
                            dim=2,
                        ).expand(-1, encoding_pooled.shape[1], -1)
                        templ_raw_enc = template_memory["encoding_pooled"][:, :1, sep_splice_ix:].expand(
                            -1, encoding_pooled.shape[1], -1
                        )
                        raw_encoding_pooled = torch.cat(
                            [raw_encoding_pooled[:, :, :sep_splice_ix], templ_raw_enc], dim=2
                        )

                if self.config.bottleneck.get("separation_loss_weight", 0) > 0:
                    if "loss" not in memory:
                        memory["loss"] = 0

                    # TODO: there must be a cleaner way of flipping the tensor ranges here...
                    if self.config.bottleneck.get("cos_separation_loss", True):
                        diff1 = cos_sim(
                            encoding_pooled[:, :, sep_splice_ix:], template_encoding_pooled[:, :, sep_splice_ix:]
                        )
                        diff2 = cos_sim(
                            encoding_pooled[:, :, :sep_splice_ix], template_encoding_pooled[:, :, :sep_splice_ix]
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
                                encoding_pooled[:, :, sep_splice_ix:] - template_encoding_pooled[:, :, sep_splice_ix:]
                            )
                            similarity_loss = (diff1 ** 2).mean(dim=-1).mean(dim=-1)

                            diff2 = (
                                encoding_pooled[:, :, :sep_splice_ix] - template_encoding_pooled[:, :, :sep_splice_ix]
                            )
                            dissimilarity_loss = torch.log(1 + 1 / (1e-2 + (diff2) ** 2)).mean(dim=-1).mean(dim=-1)
                        else:
                            diff1 = (
                                encoding_pooled[:, :, :sep_splice_ix] - template_encoding_pooled[:, :, :sep_splice_ix]
                            )
                            similarity_loss = (diff1 ** 2).mean(dim=-1).mean(dim=-1)

                            diff2 = (
                                encoding_pooled[:, :, sep_splice_ix:] - template_encoding_pooled[:, :, sep_splice_ix:]
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
            if sep_splice_ix > 0:
                memory["sep_encoding_1"] = (
                    raw_encoding_pooled[:, :, :sep_splice_ix].max(dim=1, keepdim=True).values.detach()
                )
            memory["sep_encoding_2"] = (
                raw_encoding_pooled[:, :, sep_splice_ix:].max(dim=1, keepdim=True).values.detach()
            )

        # Fwd pass through decoder block
        logits, memory = self.seq_decoder(output, memory)

        return logits, memory
