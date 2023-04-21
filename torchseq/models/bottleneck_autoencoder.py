from typing import Optional
import torch
import torch.nn as nn

from torchseq.models.encoder import SequenceEncoder
from torchseq.models.decoder import SequenceDecoder
from torchseq.models.bottleneck import PoolingBottleneck
from torchseq.models.modular_bottleneck import ModularBottleneck
from torchseq.models.contrastive_loss import ContrastiveLoss
from torchseq.utils.logging import Logger

# from torchseq.utils.functions import cos_sim, onehot
# from torchseq.models.lr_schedule import get_hyperbolic_schedule, get_tanh_schedule
# from torchseq.models.vq_code_predictor import VQCodePredictor
from torchseq.models.pooling import MultiHeadedPooling


class BottleneckAutoencoderModel(nn.Module):
    contrastive_loss: Optional[ContrastiveLoss]
    seq_encoder: SequenceEncoder
    seq_decoder: SequenceDecoder

    def __init__(self, config, input_tokenizer, output_tokenizer, src_field="source"):
        super().__init__()
        self.config = config
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        num_heads = self.config.bottleneck.get("num_heads", self.config.encoder.get("num_heads", 1))
        self.src_field = src_field

        self.seq_encoder = SequenceEncoder(
            global_config=config,
            encoder_config=config.encoder,
            tokenizer=self.input_tokenizer,
            freeze_embeddings=config.encoder.get("freeze_embeddings", config.get("freeze_embeddings", False)),
        )

        self.seq_decoder = SequenceDecoder(config, self.output_tokenizer)

        if self.config.training.get("contrastive_loss", None) is not None:
            self.contrastive_loss = ContrastiveLoss(
                metric=self.config.training.contrastive_loss.get("metric", "euclidean"),
                loss_type=self.config.training.contrastive_loss.get("loss_type", "softnn"),
                tau=self.config.training.contrastive_loss.get("tau", 1.0),
            )
        else:
            self.contrastive_loss = None

        if self.config.encoder.get("freeze", False):
            for p in self.seq_encoder.parameters():
                p.requires_grad = False
        if self.config.decoder.get("freeze", False):
            for p in self.seq_decoder.parameters():
                p.requires_grad = False

        if self.config.get("bottleneck", None) is not None:
            if config.bottleneck.get("modular", False):
                self.bottleneck = ModularBottleneck(config)
            else:
                self.bottleneck = PoolingBottleneck(config)

            if self.config.bottleneck.get("reduce_fn", "max") == "pool":
                if self.config.bottleneck.get(
                    "code_predictor", None
                ) is not None and not self.config.bottleneck.code_predictor.get("sem_only", False):
                    raise Exception("If pooling is used as the reduce_fn, sem_only must be set to true!")

                pool_dim = self.config.bottleneck.get("reduce_fn_dim", self.config.bottleneck.embedding_dim)

                self.reduce_pooling = MultiHeadedPooling(
                    self.config.bottleneck.get("reduce_fn_heads", 1),
                    pool_dim,
                    dropout=self.config.dropout,
                    use_final_linear=False,
                )

    def reduce_fn(self, inputs, mask=None):
        if self.config.bottleneck.get("reduce_fn", "max") == "pool":
            if self.reduce_proj is not None:
                inputs = self.reduce_proj(inputs)
            reduced = self.reduce_pooling(inputs, inputs, mask=mask).unsqueeze(1)
        else:
            reduced = inputs.max(dim=1).values.unsqueeze(1)
        return reduced

    def forward(self, batch, output, memory=None, tgt_field=None):
        if memory is None:
            memory = {}

        if "forced_encoding" in batch:
            memory["encoding"] = batch["forced_encoding"]
            memory["encoding_mask"] = None  # TODO: This won't work for forced encodings longer than 1!

            # print(batch['forced_encoding'].shape)

        # First pass? Construct the encoding
        if "encoding" not in memory:
            encoding, memory = self.seq_encoder(
                batch[self.src_field],
                batch[self.src_field + "_len"],
                memory,
                include_position=self.config.encoder.get("position_embeddings", True),
            )

            memory["seq_encoding"] = encoding.detach()

            if self.config.get("bottleneck", None) is not None:
                encoding_pooled, memory = self.bottleneck(
                    encoding,
                    memory,
                    batch["_global_step"],
                    head_mask=batch.get("head_mask", None),
                    forced_codes=batch.get("forced_codes", None),
                    residual_mask=batch.get("residual_mask", None),
                )
            else:
                encoding_pooled = encoding

            memory["seq_encoding_postbn"] = encoding_pooled.detach()

            prebn_encoding_pooled = memory["encoding_pooled"]

            memory["encoding"] = encoding_pooled

            # TODO: Does this belong in the Agent? (yes)
            if self.contrastive_loss is not None and self.src_field + "_group" in batch:
                # print(batch['source_text'])
                # print(len(batch['source_text']))
                # print(len(batch['source']))
                # print(len(batch['target_text']))
                # print(len(batch['target']))
                cont_loss = self.contrastive_loss(encoding_pooled, batch[self.src_field + "_group"])

                if self.training:
                    Logger().log_scalar("train/contrastive_loss", cont_loss.mean(), batch["_global_step"])
                memory["loss"] += cont_loss * self.config.training.contrastive_loss.get("weight", 1.0)

        # Fwd pass through decoder block
        logits, memory = self.seq_decoder(output, memory)

        return logits, memory
