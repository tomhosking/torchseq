import math

import torch
import torch.nn as nn
from transformers import BartModel, BertModel

from models.pooling import MultiHeadedPooling
from models.positional_embeddings import PositionalEncoding
from models.multihead_output import MultiHeadOutput
from utils.tokenizer import BPE


class TransformerParaphraseModel(nn.Module):
    def __init__(self, config, src_field="s1", loss=None):
        super().__init__()
        self.config = config

        self.src_field = src_field

        self.loss = loss

        # Embedding layers
        # self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(BPE.embeddings), freeze=config.freeze_embeddings).cpu() # TODO: this should come from a config
        self.embeddings = nn.Embedding(config.prepro.vocab_size, config.raw_embedding_dim).cpu()
        self.embeddings.weight.data = BPE.instance().embeddings
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

        decoder_layer = nn.TransformerDecoderLayer(
            config.embedding_dim,
            nhead=config.encdec.num_heads,
            dim_feedforward=config.encdec.dim_feedforward,
            dropout=config.dropout,
            activation=config.encdec.activation,
        )
        decoder_norm = nn.LayerNorm(config.embedding_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.encdec.num_decoder_layers, decoder_norm)

        # self.output_projection = nn.Linear(config.embedding_dim, config.prepro.vocab_size, bias=False).cpu()
        # # Init output projection layer with embedding matrix
        # if config.embedding_dim == config.raw_embedding_dim:
        #     self.output_projection.weight.data = self.embeddings.weight.data
        # self.output_projection.weight.requires_grad = not config.freeze_projection

        if config.embedding_dim == config.raw_embedding_dim and config.data.get("init_projection", True):
            projection_init = self.embeddings.weight.data
        else:
            projection_init = None

        if config.data.get("output_projection_heads", 1) > 1 or self.config.data.get("variational_projection", False):
            self.output_projection = MultiHeadOutput(
                config.embedding_dim,
                config.prepro.vocab_size,
                num_heads=config.data.get("output_projection_heads", 1),
                projection_init=projection_init,
                freeze_projection=config.freeze_projection,
                variational=self.config.data.get("variational_projection", False),
            )
        else:
            self.output_projection = nn.Linear(config.embedding_dim, config.prepro.vocab_size, bias=False).cpu()
            # Init output projection layer with embedding matrix
            if projection_init is not None:
                self.output_projection.weight.data = projection_init
            self.output_projection.weight.requires_grad = not config.freeze_projection

        self.encoder_pooling = MultiHeadedPooling(
            config.encdec.num_heads,
            config.embedding_dim,
            dropout=config.dropout,
            model_dim_out=config.embedding_dim,
            use_final_linear=False,
        )

        # Extra modules for a variational bottleneck
        if self.config.encdec.data.get("variational", False):
            self.encoder_logvar_pooling = MultiHeadedPooling(
                config.encdec.num_heads,
                config.embedding_dim,
                dropout=config.dropout,
                model_dim_out=config.embedding_dim,
                use_final_linear=False,
            )

        # Position encoding
        self.positional_embeddings_enc = PositionalEncoding(config.embedding_dim)
        self.positional_embeddings_dec = PositionalEncoding(config.embedding_dim)

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
        if memory is None:
            src_mask = (
                torch.FloatTensor(max_ctxt_len, max_ctxt_len)
                .fill_(float("-inf") if self.config.directional_masks else 0.0)
                .to(self.device)
            )
            src_mask = torch.triu(src_mask, diagonal=1)

            if self.config.encdec.data.get("attention_limit", None) is not None:
                src_mask = torch.tril(src_mask, diagonal=self.config.encdec.data.get("attention_limit", 0))

            context_mask = (
                torch.arange(max_ctxt_len)[None, :].cpu() >= batch[self.src_field + "_len"][:, None].cpu()
            ).to(self.device)

            ctxt_toks_embedded = self.embeddings(batch[self.src_field]).to(self.device)

            # Build the context
            if self.config.raw_embedding_dim != self.config.embedding_dim:
                ctxt_toks_embedded = self.embedding_projection(ctxt_toks_embedded)

            ctxt_embedded = ctxt_toks_embedded * math.sqrt(self.config.embedding_dim)

            ctxt_embedded = self.positional_embeddings_enc(ctxt_embedded.permute(1, 0, 2))

            #  Fwd pass through encoder
            if self.config.encdec.bert_encoder:

                # BERT expects a mask that's 1 unmasked, 0 for masked
                bert_context_mask = (~context_mask).double()

                bert_typeids = {}

                if "bart" in self.config.encdec.bert_model:
                    bert_context_mask = (1.0 - bert_context_mask.long()) * -10000.0

                self.bert_encoding = self.bert_encoder(
                    input_ids=batch[self.src_field].to(self.device), attention_mask=bert_context_mask, **bert_typeids
                )[
                    0
                ]  # , token_type_ids=batch['a_pos'].to(self.device)

                if "bart" in self.config.encdec.bert_model:
                    self.bert_encoding = self.bert_encoding.permute(1, 0, 2)

                if self.config.encdec.num_encoder_layers > 0:
                    encoding = (
                        self.encoder(
                            self.bert_encoding.permute(1, 0, 2), mask=src_mask, src_key_padding_mask=context_mask
                        )
                        .permute(1, 0, 2)
                        .contiguous()
                    )
                else:
                    encoding = self.bert_encoding

            else:
                encoding = (
                    self.encoder(ctxt_embedded, mask=src_mask, src_key_padding_mask=context_mask)
                    .permute(1, 0, 2)
                    .contiguous()
                )

            if self.config.encdec.data.get("residual", False):
                encoding = self.encoder_projection(torch.cat([encoding, ctxt_embedded.permute(1, 0, 2)], dim=-1))

            memory = (
                self.encoder_pooling(key=encoding, value=encoding).unsqueeze(1)
                if self.config.encdec.data.get("pooling", True)
                else encoding
            )

            if self.config.encdec.data.get("variational", False):
                self.mu = memory
                self.logvar = self.encoder_logvar_pooling(key=encoding, value=encoding).unsqueeze(1)

                def reparameterize(mu, logvar):
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + eps * std * self.config.encdec.data.get("prior_var_weight", 1.0)

                memory = reparameterize(self.mu, self.logvar)

            # memory = encoding

        # Build some masks
        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float("-inf")).to(self.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)

        if self.config.encdec.data.get("attention_limit", None) is not None:
            tgt_mask = torch.tril(tgt_mask, diagonal=self.config.encdec.data.get("attention_limit", 0))

        # ie how many indices are non-pad
        output_len = torch.sum(torch.ne(output, BPE.pad_id), dim=-1)

        output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() >= output_len[:, None].cpu()).to(self.device)[
            :, :output_max_len
        ]

        # Embed the output so far, then do a decoder fwd pass
        output_embedded = self.embeddings(output).to(self.device) * math.sqrt(self.config.embedding_dim)

        if self.config.raw_embedding_dim != self.config.embedding_dim:
            output_embedded = self.embedding_projection(output_embedded)

        # For some reason the Transformer implementation expects seq x batch x feat - this is weird, so permute the input and the output
        output_embedded = self.positional_embeddings_dec(output_embedded.permute(1, 0, 2))

        output = self.decoder(
            output_embedded, memory.permute(1, 0, 2), tgt_mask=tgt_mask, tgt_key_padding_mask=output_pad_mask
        ).permute(1, 0, 2)

        logits = self.output_projection(output)

        if self.config.data.get("variational_projection", False):
            self.mu, self.logvar = self.output_projection.mu, self.output_projection.logvar

        return logits, memory
