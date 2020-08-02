import math

import torch
import torch.nn as nn
from transformers import BartModel, BertModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.positional_embeddings import PositionalEncoding
from torchseq.models.multihead_output import MultiHeadOutput
from torchseq.utils.tokenizer import Tokenizer
from torchseq.models.vq_vae import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerMultiHead


class SequenceDecoder(nn.Module):
    def __init__(self, config, embeddings=None):
        super().__init__()
        self.config = config

        # Embedding layers
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = nn.Embedding(config.prepro.vocab_size, config.raw_embedding_dim).cpu()
            self.embeddings.weight.data = Tokenizer().get_embeddings(config.prepro.tokenizer)
            self.embeddings.weight.requires_grad = not config.freeze_embeddings

        decoder_layer = nn.TransformerDecoderLayer(
            config.embedding_dim,
            nhead=config.encdec.num_heads,
            dim_feedforward=config.encdec.dim_feedforward,
            dropout=config.dropout,
            activation=config.encdec.activation,
        )
        decoder_norm = nn.LayerNorm(config.embedding_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.encdec.num_decoder_layers, decoder_norm)

        if config.embedding_dim == config.raw_embedding_dim and config.data.get("init_projection", True):
            projection_init = Tokenizer().get_embeddings(config.prepro.tokenizer)
        else:
            projection_init = None

        if (
            config.data.get("output_projection_heads", 1) > 1
            or config.data.get("variational_projection", False)
            or config.data.get("normed_projection", False)
            or config.data.get("output_projection_embeddings", 1) > 1
        ):
            self.output_projection = MultiHeadOutput(
                config.embedding_dim,
                config.prepro.vocab_size,
                num_heads=config.data.get("output_projection_heads", 1),
                num_projections=config.data.get("output_projection_embeddings", 1),
                projection_init=projection_init,
                freeze_projection=config.freeze_projection,
                variational=self.config.data.get("variational_projection", False),
                normed=self.config.data.get("normed_projection", False),
            ).cpu()
        else:
            self.output_projection = nn.Linear(config.embedding_dim, config.prepro.vocab_size, bias=False).cpu()
            # Init output projection layer with embedding matrix
            if projection_init is not None:
                self.output_projection.weight.data = projection_init
            self.output_projection.weight.requires_grad = not config.freeze_projection

        self.positional_embeddings = PositionalEncoding(config.embedding_dim)

    def forward(self, output_seq, memory):

        output_max_len = output_seq.size()[-1]

        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float("-inf")).to(output_seq.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)

        if self.config.encdec.data.get("attention_limit", None) is not None:
            tgt_mask = torch.tril(tgt_mask, diagonal=self.config.encdec.data.get("attention_limit", 0))

        # ie how many indices are non-pad
        output_len = torch.sum(torch.ne(output_seq, Tokenizer().pad_id), dim=-1)

        output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() >= output_len[:, None].cpu()).to(
            output_seq.device
        )[:, :output_max_len]

        # Embed the output so far
        output_embedded = self.embeddings(output_seq).to(output_seq.device) * math.sqrt(self.config.embedding_dim)

        if self.config.raw_embedding_dim != self.config.embedding_dim:
            output_embedded = self.embedding_projection(output_embedded)

        # For some reason the Transformer implementation expects seq x batch x feat - this is weird, so permute the input and the output
        output_embedded = self.positional_embeddings(output_embedded.permute(1, 0, 2))

        # Decoder block fwd pass
        output_seq = self.decoder(
            output_embedded,
            memory["encoding"].permute(1, 0, 2),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=output_pad_mask,
            memory_key_padding_mask=memory["encoding_mask"],
        ).permute(1, 0, 2)

        # Embeddings -> logits
        if self.config.data.get("variational_projection", False):
            logits, memory = self.output_projection(output_seq)
        else:
            logits = self.output_projection(output_seq)

        return logits, memory
