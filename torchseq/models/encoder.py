import math

import torch
import torch.nn as nn
from transformers import BartModel, BertModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.positional_embeddings import PositionalEncoding
from torchseq.models.multihead_output import MultiHeadOutput
from torchseq.utils.tokenizer import Tokenizer
from torchseq.models.vq_vae import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerMultiHead


class SequenceEncoder(nn.Module):
    def __init__(self, config, embeddings=None):
        super().__init__()
        self.config = config

        # Embedding layers
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = nn.Embedding(config.prepro.vocab_size, config.raw_embedding_dim).cpu()
            self.embeddings.weight.data = Tokenizer().get_embeddings(config.encdec.bert_model)
            self.embeddings.weight.requires_grad = not config.freeze_embeddings

        self.embedding_projection = nn.utils.weight_norm(
            nn.Linear(config.raw_embedding_dim, config.embedding_dim, bias=False)
        )

        # Encoder/decoders
        if config.encdec.bert_encoder:

            if "bart" in config.encdec.bert_model:
                bart_model = BartModel.from_pretrained(config.encdec.bert_model)
                self.bert_encoder = bart_model.encoder
                del bart_model.decoder
            else:
                self.bert_encoder = BertModel.from_pretrained(config.encdec.bert_model)

        if self.config.encdec.data.get("residual", False):
            self.encoder_projection = nn.utils.weight_norm(
                nn.Linear(config.embedding_dim * 2, config.embedding_dim, bias=False)
            )

        encoder_layer = nn.TransformerEncoderLayer(
            config.embedding_dim,
            nhead=config.encdec.num_heads,
            dim_feedforward=config.encdec.dim_feedforward,
            dropout=config.dropout,
            activation=config.encdec.activation,
        )
        encoder_norm = nn.LayerNorm(config.embedding_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encdec.num_encoder_layers, encoder_norm)

        # Position encoding
        self.positional_embeddings = PositionalEncoding(config.embedding_dim)

    def forward(self, input_seq, input_seq_len, memory):
        max_input_len = input_seq.shape[1]

        # Re-normalise the projections...
        with torch.no_grad():
            self.embedding_projection.weight_g.div_(self.embedding_projection.weight_g)
            if self.config.encdec.data.get("residual", False):
                self.encoder_projection.weight_g.div_(self.encoder_projection.weight_g)

        # Set up some masks
        src_mask = (
            torch.FloatTensor(max_input_len, max_input_len)
            .fill_(float("-inf") if self.config.directional_masks else 0.0)
            .to(input_seq.device)
        )
        src_mask = torch.triu(src_mask, diagonal=1)

        if self.config.encdec.data.get("attention_limit", None) is not None:
            src_mask = torch.tril(src_mask, diagonal=self.config.encdec.data.get("attention_limit", 0))

        padding_mask = (torch.arange(max_input_len)[None, :].cpu() >= input_seq_len[:, None].cpu()).to(
            input_seq.device
        )

        # Embed the input
        input_toks_embedded = self.embeddings(input_seq).to(input_seq.device)

        if self.config.raw_embedding_dim != self.config.embedding_dim:
            input_toks_embedded = self.embedding_projection(input_toks_embedded)

        input_embedded = input_toks_embedded * math.sqrt(self.config.embedding_dim)

        input_embedded = self.positional_embeddings(input_embedded.permute(1, 0, 2))

        #  Fwd pass through encoder
        if self.config.encdec.bert_encoder:

            # BERT expects a mask that's 1 unmasked, 0 for masked
            bert_padding_mask = (~padding_mask).double()

            bert_typeids = {}

            if "bart" in self.config.encdec.bert_model:
                bert_padding_mask = (1.0 - bert_padding_mask.long()) * -10000.0

            self.bert_encoding = self.bert_encoder(
                input_ids=input_seq.to(input_seq.device), attention_mask=bert_padding_mask, **bert_typeids
            )[0]

            if "bart" in self.config.encdec.bert_model:
                self.bert_encoding = self.bert_encoding.permute(1, 0, 2)

            if self.config.encdec.num_encoder_layers > 0:
                encoding = (
                    self.encoder(self.bert_encoding.permute(1, 0, 2), mask=src_mask, src_key_padding_mask=padding_mask)
                    .permute(1, 0, 2)
                    .contiguous()
                )
            else:
                encoding = self.bert_encoding

        else:
            encoding = (
                self.encoder(input_embedded, mask=src_mask, src_key_padding_mask=padding_mask)
                .permute(1, 0, 2)
                .contiguous()
            )

        # Include original input?
        if self.config.encdec.data.get("residual", False):
            encoding = self.encoder_projection(torch.cat([encoding, input_embedded.permute(1, 0, 2)], dim=-1))

        return encoding, memory
