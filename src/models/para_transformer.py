import math

import torch
import torch.nn as nn
from transformers import BertModel, BartModel

from models.pooling import MultiHeadedPooling
from models.positional_embeddings import PositionalEncoding
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

        self.output_projection = nn.Linear(config.embedding_dim, config.prepro.vocab_size, bias=False).cpu()

        self.encoder_pooling = MultiHeadedPooling(
            config.encdec.num_heads,
            config.embedding_dim,
            dropout=config.dropout,
            model_dim_out=config.embedding_dim,
            use_final_linear=False,
        )

        # Init output projection layer with embedding matrix
        if config.embedding_dim == config.raw_embedding_dim:
            self.output_projection.weight.data = self.embeddings.weight.data
        self.output_projection.weight.requires_grad = not config.freeze_projection

        # Position encoding
        self.positional_embeddings_enc = PositionalEncoding(config.embedding_dim)
        self.positional_embeddings_dec = PositionalEncoding(config.embedding_dim)

    def forward(self, batch, output, memory=None, tgt_field=None):

        # Re-normalise the projections...
        with torch.no_grad():
            self.embedding_projection.weight_g.div_(self.embedding_projection.weight_g)
            if self.config.encdec.data.get("residual", False):
                self.encoder_projection.weight_g.div_(self.encoder_projection.weight_g)

        # print(BPE.decode(batch['a'][0][:batch['a_len'][0]]), [BPE.instance().decode([x.item()])  for i,x in enumerate(batch['c'][0]) if batch['a_pos'][0][i].item() > 0], BPE.decode(batch['q'][0][:batch['q_len'][0]]))
        # print([BPE.instance().decode([x.item()])+'/'+str(batch['a_pos'][0][i].item())  for i,x in enumerate(batch['c'][0])])
        # exit()

        # Get some sizes
        max_ctxt_len = batch[self.src_field].shape[1]
        # max_q_len = torch.max(batch['q_len'])
        curr_batch_size = batch[self.src_field].size()[0]
        output_max_len = output.size()[-1]

        # First pass? Construct the encoding
        if memory is None:
            src_mask = (
                torch.FloatTensor(max_ctxt_len, max_ctxt_len)
                .fill_(float("-inf") if self.config.directional_masks else 0.0)
                .to(self.device)
            )
            src_mask = torch.triu(src_mask, diagonal=1)
            # src_mask = src_mask.where(batch['a_pos'] > 0, torch.zeros_like(src_mask).unsqueeze(-1))

            context_mask = (
                torch.arange(max_ctxt_len)[None, :].cpu() >= batch[self.src_field + "_len"][:, None].cpu()
            ).to(self.device)

            ctxt_toks_embedded = self.embeddings(batch[self.src_field]).to(self.device)

            # Build the context
            if self.config.raw_embedding_dim != self.config.embedding_dim:
                ctxt_toks_embedded = self.embedding_projection(ctxt_toks_embedded)

            ctxt_embedded = ctxt_toks_embedded * math.sqrt(self.config.embedding_dim)

            ctxt_embedded = self.positional_embeddings_enc(ctxt_embedded.permute(1, 0, 2))

            # encoding = self.encoder(ctxt_embedded, mask=src_mask, src_key_padding_mask=context_mask).permute(1,0,2).contiguous()
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
                # print(encoding.shape)
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

            # memory = encoding

        # Build some masks
        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float("-inf")).to(self.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)

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

        if tgt_field is not None:
            bos_logits = (
                torch.FloatTensor(curr_batch_size, 1, self.config.prepro.vocab_size)
                .fill_(float("-1e18"))
                .to(self.device)
            )
            bos_logits[:, :, BPE.bos_id] = float("1e18")
            loss_logits = torch.cat([bos_logits, logits], dim=1)
            loss = self.loss(loss_logits.permute(0, 2, 1), batch[tgt_field])
        else:
            loss = None

        return logits, memory, loss
