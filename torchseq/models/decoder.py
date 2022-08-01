import math

import torch
import torch.nn as nn

from torchseq.models.positional_embeddings import PositionalEncoding
from torchseq.models.multihead_output import MultiHeadOutput
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.functions import initialize_truncated_normal_


class SequenceDecoder(nn.Module):
    def __init__(self, config, tokenizer, embeddings=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Embedding layers
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = nn.Embedding(
                tokenizer.vocab_size,
                config.get_first(["output_raw_embedding_dim", "raw_embedding_dim"]),
            )
            if self.tokenizer.has_embeddings:
                self.embeddings.weight.data = self.tokenizer.get_embeddings()
            else:
                torch.nn.init.xavier_uniform_(self.embeddings.weight.data, gain=1.0)
                # initialize_truncated_normal_(self.embeddings.weight.data, std=0.02)
            self.embeddings.weight.requires_grad = not (
                config.decoder.get("freeze_embeddings", config.get("freeze_embeddings", False))
                if "decoder" in config.data
                else config.get("freeze_embeddings", False)
            )
            self.embeddings.cpu()
            self.embeddings.force_device = True

        decoder_layer = nn.TransformerDecoderLayer(
            config.decoder.embedding_dim,
            nhead=config.encdec.num_heads,
            dim_feedforward=config.encdec.dim_feedforward,
            dropout=config.dropout,
            activation=config.encdec.activation,
        )
        decoder_norm = nn.LayerNorm(config.decoder.embedding_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.encdec.num_decoder_layers, decoder_norm)

        projection_init = None
        if (
            config.decoder.embedding_dim == config.get_first(["output_raw_embedding_dim", "raw_embedding_dim"])
            and config.data.get("init_projection", True)
            and self.tokenizer.has_embeddings
        ):
            projection_init = self.tokenizer.get_embeddings()

        if (
            config.data.get("output_projection_heads", 1) > 1
            or config.data.get("variational_projection", False)
            or config.data.get("normed_projection", False)
            or config.data.get("output_projection_embeddings", 1) > 1
        ):
            self.output_projection = MultiHeadOutput(
                config.decoder.embedding_dim,
                config.prepro.get_first(["output_vocab_size", "vocab_size"]),
                num_heads=config.data.get("output_projection_heads", 1),
                num_projections=config.data.get("output_projection_embeddings", 1),
                projection_init=projection_init,
                freeze_projection=config.freeze_projection,
                variational=self.config.data.get("variational_projection", False),
                normed=self.config.data.get("normed_projection", False),
            ).cpu()
        else:
            self.output_projection = nn.Linear(
                config.decoder.embedding_dim,
                config.prepro.get_first(["output_vocab_size", "vocab_size"]),
                bias=False,
            ).cpu()
            # Init output projection layer with embedding matrix
            if projection_init is not None:
                self.output_projection.weight.data = projection_init
            self.output_projection.weight.requires_grad = not config.freeze_projection

        self.positional_embeddings = PositionalEncoding(config.decoder.embedding_dim)

    def forward(self, output_seq, memory):

        output_max_len = output_seq.size()[-1]

        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float("-inf")).to(output_seq.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)

        if self.config.encdec.data.get("attention_limit", None) is not None:
            tgt_mask = torch.tril(tgt_mask, diagonal=self.config.encdec.data.get("attention_limit", 0))

        # ie how many indices are non-pad
        output_len = torch.sum(torch.ne(output_seq, self.tokenizer.pad_id), dim=-1)

        output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() >= output_len[:, None].cpu()).to(
            output_seq.device
        )[:, :output_max_len]

        # Embed the output so far
        output_embedded = self.embeddings(output_seq).to(output_seq.device) * math.sqrt(
            self.config.decoder.embedding_dim
        )

        # if self.config.raw_embedding_dim != self.config.decoder.embedding_dim:
        #     output_embedded = self.embedding_projection(output_embedded)

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
