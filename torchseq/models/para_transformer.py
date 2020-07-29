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


class TransformerParaphraseModel(nn.Module):
    def __init__(self, config, src_field="s1"):
        super().__init__()
        self.config = config

        self.src_field = src_field

        self.seq_encoder = SequenceEncoder(config)
        self.bottleneck = PoolingBottleneck(config)
        self.seq_decoder = SequenceDecoder(config, embeddings=self.seq_encoder.embeddings)

    def forward(self, batch, output, memory=None, tgt_field=None):
        if memory is None:
            memory = dict()

        # First pass? Construct the encoding
        if "encoding" not in memory:
            encoding, memory = self.seq_encoder(batch[self.src_field], batch[self.src_field + "_len"], memory)
            encoding_pooled, memory = self.bottleneck(encoding, memory)

            if "template" in batch:
                template_memory = {}
                template_encoding, template_memory = self.seq_encoder(
                    batch["template"], batch["template_len"], template_memory
                )
                template_encoding_pooled, template_memory = self.bottleneck(template_encoding, template_memory)
                # TODO: splice the template encoding into the input

                splice_ix = (
                    self.config.embedding_dim
                    // self.config.encdec.get("quantizer_heads", 1)
                    * self.config.encdec.get("quantizer_num_residual", 0)
                )

                seq_encoding = encoding_pooled[:, :, :splice_ix]
                template_encoding = encoding_pooled[:, :, splice_ix:]

                encoding_pooled = torch.cat([seq_encoding, template_encoding], dim=-1)

            memory["encoding"] = encoding_pooled
            memory["encoding_mask"] = None

        # Fwd pass through decoder block
        logits, memory = self.seq_decoder(output, memory)

        return logits, memory
