import math

import torch
import torch.nn as nn
from transformers import BartModel, BertModel

from models.pooling import MultiHeadedPooling
from models.positional_embeddings import PositionalEncoding
from models.multihead_output import MultiHeadOutput
from utils.tokenizer import BPE


class TransformerLanguageModel(nn.Module):
    def __init__(self, config, src_field="s1"):
        super().__init__()
        self.config = config

        self.src_field = src_field

        # Embedding layers
        # self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(BPE.embeddings), freeze=config.freeze_embeddings).cpu() # TODO: this should come from a config
        self.embeddings = nn.Embedding(config.prepro.vocab_size, config.raw_embedding_dim).cpu()
        self.embeddings.weight.data = BPE.instance().embeddings
        self.embeddings.weight.requires_grad = not config.freeze_embeddings

        self.embedding_projection = nn.utils.weight_norm(
            nn.Linear(config.raw_embedding_dim, config.embedding_dim, bias=False)
        )

        # Encoder

        encoder_layer = nn.TransformerEncoderLayer(
            config.embedding_dim,
            nhead=config.encdec.num_heads,
            dim_feedforward=config.encdec.dim_feedforward,
            dropout=config.dropout,
            activation=config.encdec.activation,
        )
        encoder_norm = nn.LayerNorm(config.embedding_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encdec.num_encoder_layers, encoder_norm)

        # self.output_projection = nn.Linear(config.embedding_dim, config.prepro.vocab_size, bias=False).cpu()
        # # Init output projection layer with embedding matrix
        # if config.embedding_dim == config.raw_embedding_dim:
        #     self.output_projection.weight.data = self.embeddings.weight.data
        # self.output_projection.weight.requires_grad = not config.freeze_projection

        if config.embedding_dim == config.raw_embedding_dim and config.data.get("init_projection", True):
            projection_init = self.embeddings.weight.data
        else:
            projection_init = None
        self.output_projection = MultiHeadOutput(
            config.embedding_dim,
            config.prepro.vocab_size,
            num_heads=config.data.get("output_projection_heads", 1),
            projection_init=projection_init,
            freeze_projection=config.freeze_projection,
            variational=self.config.data.get("variational_projection", False),
        )

        # Position encoding
        self.positional_embeddings_enc = PositionalEncoding(config.embedding_dim)

    def forward(self, batch, output, memory=None, tgt_field=None):

        # Re-normalise the projections...
        with torch.no_grad():
            self.embedding_projection.weight_g.div_(self.embedding_projection.weight_g)
            if self.config.encdec.data.get("residual", False):
                self.encoder_projection.weight_g.div_(self.encoder_projection.weight_g)

        # Get some sizes
        max_ctxt_len = batch[self.src_field].shape[1]
        output_max_len = output.size()[-1]

        # First pass? Construct the encoding

        src_mask = (
            torch.FloatTensor(max_ctxt_len, max_ctxt_len)
            .fill_(float("-inf") if self.config.directional_masks else 0.0)
            .to(self.device)
        )
        src_mask = torch.triu(src_mask, diagonal=1)

        context_mask = (torch.arange(max_ctxt_len)[None, :].cpu() >= batch[self.src_field + "_len"][:, None].cpu()).to(
            self.device
        )

        ctxt_toks_embedded = self.embeddings(batch[self.src_field]).to(self.device)

        # Build the context
        if self.config.raw_embedding_dim != self.config.embedding_dim:
            ctxt_toks_embedded = self.embedding_projection(ctxt_toks_embedded)

        ctxt_embedded = ctxt_toks_embedded * math.sqrt(self.config.embedding_dim)

        ctxt_embedded = self.positional_embeddings_enc(ctxt_embedded.permute(1, 0, 2))

        encoding = (
            self.encoder(ctxt_embedded, mask=src_mask, src_key_padding_mask=context_mask).permute(1, 0, 2).contiguous()
        )

        logits = self.output_projection(encoding)

        if self.config.data.get("variational_projection", False):
            self.mu, self.logvar = self.output_projection.mu, self.output_projection.logvar

        return logits, memory
