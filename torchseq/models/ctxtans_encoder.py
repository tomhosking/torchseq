from typing import Dict, Union, Tuple
import math
import logging

import torch
import torch.nn as nn
from transformers import BartModel, BertModel, RobertaModel, MBartModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.positional_embeddings import PositionalEncoding
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.config import Config
from torchseq.utils.functions import initialize_truncated_normal_, init_bert_params

import torchseq.models.transformer as custom_transformer


class ContextAnswerEncoder(nn.Module):
    def __init__(self, config, input_tokenizer, embeddings=None, freeze_embeddings=False):
        super().__init__()
        self.config = config
        self.input_tokenizer = input_tokenizer

        # Embedding layers
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = nn.Embedding(
                config.prepro.get_first(["input_vocab_size", "vocab_size"]),
                config.get_first(["input_raw_embedding_dim", "raw_embedding_dim"]),
            ).cpu()
            if self.input_tokenizer.has_embeddings:
                self.embeddings.weight.data = self.input_tokenizer.get_embeddings()
            self.embeddings.weight.requires_grad = not freeze_embeddings

        self.embedding_projection = nn.utils.weight_norm(
            nn.Linear(
                config.get_first(["input_raw_embedding_dim", "raw_embedding_dim"]),
                config.encoder.embedding_dim,
                bias=False,
            )
        )

        self.bert_embedding_projection = nn.utils.weight_norm(
            nn.Linear(
                config.encoder.embedding_dim * 1 + config.bio_embedding_dim, config.encoder.embedding_dim, bias=False
            )
        )

        self.bio_embeddings = (
            nn.Embedding.from_pretrained(torch.eye(config.bio_embedding_dim), freeze=True).cpu()
            if config.onehot_bio
            else nn.Embedding(3, config.bio_embedding_dim).cpu()
        )

        # Encoder/decoders
        self.pretrained_model_slug = None
        if config.encoder.get("bert_encoder", False) or config.encoder.get("pretrained_encoder", None) is not None:
            self.pretrained_model_slug = (
                config.encoder.pretrained_encoder
                if config.encoder.get("pretrained_encoder", None) is not None
                else config.encoder.bert_model
            )

            self.freeze_bert = not self.config.encoder.bert_finetune
            if "bart-" in self.pretrained_model_slug:
                bart_model = BartModel.from_pretrained(self.pretrained_model_slug)
                self.pretrained_encoder = bart_model.encoder
                del bart_model.decoder
            elif "roberta-" in self.pretrained_model_slug:
                self.pretrained_encoder = RobertaModel.from_pretrained(self.pretrained_model_slug)
            else:
                self.pretrained_encoder = BertModel.from_pretrained(self.pretrained_model_slug)
            # self.pretrained_encoder.train()
            # for param in self.pretrained_encoder.parameters():
            #     param.requires_grad = True

        if self.config.encoder.data.get("pre_ln", False):
            encoder_layer = custom_transformer.TransformerEncoderLayer(
                config.encoder.embedding_dim
                + (0 if self.pretrained_model_slug is not None else config.bio_embedding_dim),
                nhead=config.encoder.num_heads,
                dim_feedforward=config.encoder.dim_feedforward,
                dropout=config.dropout,
                activation=config.encoder.activation,
            )
            encoder_norm = nn.LayerNorm(
                config.encoder.embedding_dim
                + (0 if self.pretrained_model_slug is not None else config.bio_embedding_dim)
            )
            self.encoder = custom_transformer.TransformerEncoder(
                encoder_layer, config.encoder.num_layers, encoder_norm
            )

        else:
            encoder_layer = nn.TransformerEncoderLayer(
                config.encoder.embedding_dim
                + (0 if self.pretrained_model_slug is not None else config.bio_embedding_dim),
                nhead=config.encoder.num_heads,
                dim_feedforward=config.encoder.dim_feedforward,
                dropout=config.dropout,
                activation=config.encoder.activation,
                batch_first=True,
            )
            encoder_norm = nn.LayerNorm(
                config.encoder.embedding_dim
                + (0 if self.pretrained_model_slug is not None else config.bio_embedding_dim)
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, config.encoder.num_layers, encoder_norm, enable_nested_tensor=True
            )

            if self.config.encoder.get("init_like_bert", False):
                init_bert_params(self.encoder)

        # if self.config.encoder.data.get("xav_uni_init", False):
        #     for p in self.encoder.parameters():
        #         if p.dim() > 1:
        #             nn.init.xavier_uniform_(p)

        # Encoder combination
        num_encoder_outputs = sum(
            [1 if v else 0 for k, v in config.encoder_outputs.data.items() if k != "c_ans_labels"]
        )
        memory_dim = (
            config.encoder.embedding_dim + (0 if self.pretrained_model_slug is not None else config.bio_embedding_dim)
        ) * num_encoder_outputs
        memory_dim += self.config.bio_embedding_dim if self.config.encoder_outputs.c_ans_labels else 0
        self.encoder_projection = nn.utils.weight_norm(nn.Linear(memory_dim, config.encoder.embedding_dim, bias=False))

        # Pooling layers
        self.ans_pooling = MultiHeadedPooling(
            config.encoder.num_heads,
            config.encoder.embedding_dim + config.bio_embedding_dim,
            dropout=config.dropout,
            model_dim_out=config.encoder.embedding_dim,
            use_final_linear=False,
        )
        self.ctxt_pooling = MultiHeadedPooling(
            config.encoder.num_heads,
            config.encoder.embedding_dim + config.bio_embedding_dim,
            dropout=config.dropout,
            model_dim_out=config.encoder.embedding_dim,
            use_final_linear=False,
            use_bilinear=True,
        )

        # Position encoding
        self.positional_embeddings_enc = PositionalEncoding(
            config.encoder.embedding_dim + (0 if self.pretrained_model_slug is not None else config.bio_embedding_dim)
        )

        if self.config.encoder.get("memory_tokens", 0) > 0:
            w = torch.empty(
                self.config.encoder.get("memory_tokens", 0),
                config.encoder.embedding_dim
                + (0 if self.pretrained_model_slug is not None else config.bio_embedding_dim),
            )
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("relu"))
            self.memory_tokens = nn.Parameter(w, requires_grad=True)

    def forward(self, ctxt_seq, ctxt_seq_len, a_pos, memory):
        # Re-normalise the projections...
        with torch.no_grad():
            self.embedding_projection.weight_g.div_(self.embedding_projection.weight_g)
            self.bert_embedding_projection.weight_g.div_(self.bert_embedding_projection.weight_g)
            self.encoder_projection.weight_g.div_(self.encoder_projection.weight_g)

        # Get some sizes
        max_ctxt_len = ctxt_seq.shape[1]

        if self.config.encoder.get("memory_tokens", 0) > 0:
            max_ctxt_len += self.config.encoder.get("memory_tokens", 0)
            ctxt_seq_len += self.config.encoder.get("memory_tokens", 0)

        context_mask = (torch.arange(max_ctxt_len)[None, :].cpu() >= ctxt_seq_len[:, None].cpu()).to(ctxt_seq.device)
        memory["encoding_mask"] = context_mask

        # First pass? Construct the encoding
        if "encoding" not in memory:
            src_mask = (
                torch.FloatTensor(max_ctxt_len, max_ctxt_len)
                .fill_(-torch.inf if self.config.directional_masks else 0.0)
                .to(ctxt_seq.device)
            )
            src_mask = torch.triu(src_mask, diagonal=1)

            ctxt_toks_embedded = self.embeddings(ctxt_seq).to(ctxt_seq.device)
            ctxt_ans_embedded = self.bio_embeddings(a_pos).to(ctxt_seq.device)

            # Build the context
            if self.config.encoder.embedding_dim != self.config.get_first(
                ["input_raw_embedding_dim", "raw_embedding_dim"]
            ):
                ctxt_toks_embedded = self.embedding_projection(ctxt_toks_embedded)

            if self.pretrained_model_slug is not None:
                ctxt_embedded = ctxt_toks_embedded * math.sqrt(self.config.encoder.embedding_dim)
            else:
                ctxt_embedded = torch.cat([ctxt_toks_embedded, ctxt_ans_embedded], dim=-1) * math.sqrt(
                    self.config.encoder.embedding_dim
                )

            ctxt_embedded = self.positional_embeddings_enc(ctxt_embedded)

            if self.config.encoder.get("memory_tokens", 0) > 0:
                ctxt_embedded = torch.cat(
                    [self.memory_tokens.unsqueeze(1).expand(-1, ctxt_embedded.shape[1], -1), ctxt_embedded]
                )

            # Fwd pass through encoder
            if self.pretrained_model_slug is not None:

                # BERT expects a mask that's 1 unmasked, 0 for masked
                bert_context_mask = (~context_mask).double()

                if "bert_typeids" in self.config.encoder.data and self.config.encoder.bert_typeids:
                    bert_typeids = {"token_type_ids": a_pos.to(ctxt_seq.device)}
                else:
                    bert_typeids = {}

                self.bert_encoding = self.pretrained_encoder(
                    input_ids=ctxt_seq.to(ctxt_seq.device), attention_mask=bert_context_mask, **bert_typeids
                )[0]

                if "bart" in self.pretrained_model_slug:
                    self.bert_encoding = self.bert_encoding

                if self.config.encoder.embedding_dim != self.config.get_first(
                    ["input_raw_embedding_dim", "raw_embedding_dim"]
                ):
                    self.bert_encoding = self.embedding_projection(self.bert_encoding)

                if self.config.encoder.num_layers > 0:
                    bert_encoding_augmented = torch.cat(
                        [self.bert_encoding, ctxt_ans_embedded], dim=-1
                    )  # ctxt_embedded.permute(1,0,2)
                    bert_encoding_augmented = self.bert_embedding_projection(bert_encoding_augmented)
                    encoding = self.encoder(
                        bert_encoding_augmented, mask=src_mask, src_key_padding_mask=context_mask
                    ).contiguous()
                else:
                    encoding = self.bert_encoding

            else:
                encoding = self.encoder(ctxt_embedded, mask=src_mask, src_key_padding_mask=context_mask).contiguous()

            # Construct the encoder output by combining a few diff sources
            memory_elements = []
            ans_mask = a_pos == 0
            if self.config.encoder_outputs.c_raw:
                memory_elements.append(ctxt_embedded)

            if self.config.encoder_outputs.a_raw:
                memory_elements.append(ctxt_embedded.masked_fill(ans_mask, 0))

            if self.config.encoder_outputs.c_enc:
                memory_elements.append(encoding)

            if self.config.encoder_outputs.c_enc_pool:
                ctxt_pooled = self.ctxt_pooling(key=encoding, value=encoding).unsqueeze(1)
                memory_elements.append(ctxt_pooled.expand(-1, max_ctxt_len, -1))

            if self.config.encoder_outputs.a_enc:
                memory_elements.append(encoding.masked_fill(ans_mask, 0))

            if self.config.encoder_outputs.c_ans_labels:
                memory_elements.append(ctxt_ans_embedded)

            if self.config.encoder_outputs.a_enc_pool:
                ans_pooled = self.ans_pooling(encoding, encoding, mask=ans_mask).unsqueeze(1)
                memory_elements.append(ans_pooled.expand(-1, max_ctxt_len, -1))

            # This one needs work...
            if self.config.encoder_outputs.c_enc_anspool:
                ans_pooled = self.ans_pooling(encoding, encoding, mask=ans_mask).unsqueeze(1)
                ctxt_anspooled = self.ctxt_pooling(key=ans_pooled, value=encoding).unsqueeze(1)
                memory_elements.append(ctxt_anspooled.expand(-1, max_ctxt_len, -1))

            memory_full = torch.cat(
                memory_elements, dim=-1
            )  # , encoding, ctxt_embedded.permute(1,0,2), memory_pooled.expand(-1, max_ctxt_len, -1)

            if len(memory_elements) > 1 or True:
                memory_full = self.encoder_projection(memory_full)

        return memory_full, memory
