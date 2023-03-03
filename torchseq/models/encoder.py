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


class SequenceEncoder(nn.Module):

    global_config: Config
    encoder_config: Config
    tokenizer: Tokenizer
    embeddings: nn.Embedding
    encoder: Union[custom_transformer.TransformerEncoder, nn.TransformerEncoder]

    def __init__(
        self,
        global_config: Config,
        encoder_config: Config,
        tokenizer: Tokenizer,
        embeddings=None,
        freeze_embeddings=False,
    ):
        super().__init__()
        self.global_config = global_config
        self.encoder_config = encoder_config
        self.tokenizer = tokenizer

        # Embedding layers
        if embeddings is not None:
            self.embeddings = embeddings
            self.embeddings.force_device = True  # type: ignore # dynamic attr
        elif not encoder_config.get("bert_encoder", False) and encoder_config.get("pretrained_encoder", None) is None:
            self.embeddings = nn.Embedding(
                tokenizer.vocab_size,
                global_config.get_first(["input_raw_embedding_dim", "raw_embedding_dim"]),
            ).cpu()
            if self.tokenizer.has_embeddings and self.encoder_config.get("init_embeds_from_tokenizer", True):
                self.embeddings.weight.data = self.tokenizer.get_embeddings()
            else:

                if self.encoder_config.get("init_embeds_like_bert", False):
                    init_bert_params(self.embeddings)
                else:
                    torch.nn.init.xavier_uniform_(self.embeddings.weight.data, gain=1.0)
                    # initialize_truncated_normal_(
                    #     self.embeddings.weight.data, std=1 / math.sqrt(config.decoder.embedding_dim)
                    # )
            self.embeddings.weight.requires_grad = not freeze_embeddings
            self.embeddings.cpu()
            self.embeddings.force_device = True  # type: ignore # dynamic attr

        if self.encoder_config.embedding_dim != global_config.get_first(
            ["input_raw_embedding_dim", "raw_embedding_dim"]
        ):
            self.embedding_projection = nn.utils.weight_norm(
                nn.Linear(
                    global_config.get_first(["input_raw_embedding_dim", "raw_embedding_dim"]),
                    encoder_config.embedding_dim,
                    bias=False,
                )
            )

        # Encoder/decoders
        self.pretrained_model_slug = None
        if encoder_config.get("bert_encoder", False) or encoder_config.get("pretrained_encoder", None) is not None:
            self.pretrained_model_slug = (
                encoder_config.pretrained_encoder
                if encoder_config.get("pretrained_encoder", None) is not None
                else encoder_config.bert_model
            )
            if "mbart" in self.pretrained_model_slug:
                bart_model = MBartModel.from_pretrained(self.pretrained_model_slug)
                self.pretrained_encoder = bart_model.encoder
                del bart_model.decoder
            elif "bart" in self.pretrained_model_slug:
                bart_model = BartModel.from_pretrained(self.pretrained_model_slug)
                self.pretrained_encoder = bart_model.encoder
                del bart_model.decoder
            elif "roberta-" in self.pretrained_model_slug:
                self.pretrained_encoder = RobertaModel.from_pretrained(self.pretrained_model_slug)
            else:
                # TODO: Make this an AutoModel?
                self.pretrained_encoder = BertModel.from_pretrained(self.pretrained_model_slug)

            if encoder_config.get("freeze_pretrained", False):
                self.pretrained_encoder.requires_grad = False
        else:
            self.pretrained_encoder = None

        if self.encoder_config.get("residual", False):
            self.encoder_projection = nn.utils.weight_norm(
                nn.Linear(encoder_config.embedding_dim * 2, encoder_config.embedding_dim, bias=False)
            )
        if self.encoder_config.get("pre_residual", False):
            self.token_projection = nn.utils.weight_norm(
                nn.Linear(
                    global_config.get_first(["input_raw_embedding_dim", "raw_embedding_dim"]),
                    encoder_config.embedding_dim,
                    bias=False,
                )
            )

        if self.encoder_config.data.get("pre_ln", False):
            encoder_layer_custom = custom_transformer.TransformerEncoderLayer(
                encoder_config.embedding_dim
                + (0 if self.pretrained_model_slug is not None else global_config.bio_embedding_dim),
                nhead=encoder_config.num_heads,
                dim_feedforward=encoder_config.dim_feedforward,
                dropout=global_config.dropout,
                activation=encoder_config.activation,
            )
            encoder_norm = nn.LayerNorm(encoder_config.embedding_dim)
            self.encoder = custom_transformer.TransformerEncoder(
                encoder_layer_custom, encoder_config.num_layers, encoder_norm
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                encoder_config.embedding_dim,
                nhead=encoder_config.num_heads,
                dim_feedforward=encoder_config.dim_feedforward,
                dropout=global_config.dropout,
                activation=encoder_config.activation,
                batch_first=True,
            )
            encoder_norm = nn.LayerNorm(encoder_config.embedding_dim)
            self.encoder = nn.TransformerEncoder(
                encoder_layer, encoder_config.num_layers, encoder_norm, enable_nested_tensor=True
            )

        # Position encoding
        self.positional_embeddings = PositionalEncoding(encoder_config.embedding_dim)

    def forward(
        self,
        input_seq: torch.Tensor,
        input_seq_len: torch.Tensor,
        memory: Dict[str, torch.Tensor],
        include_position: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        max_input_len = input_seq.shape[1]

        # Set up some masks
        src_mask = torch.zeros(max_input_len, max_input_len, dtype=torch.bool).to(input_seq.device)
        # is_causal = False # ready for pt2

        if self.global_config.directional_masks:
            src_mask = src_mask.logical_not().triu_(diagonal=1)
            # is_causal = True

        if self.encoder_config.data.get("attention_limit", None) is not None:
            raise Exception("attention_limit no longer supported!")
            src_mask = torch.tril(src_mask, diagonal=self.encoder_config.data.get("attention_limit", 0))

        if self.encoder_config.data.get("no_diagonal_attn", False):
            raise Exception("no_diagonal_attn no longer supported!")
            src_mask += -torch.inf * torch.eye(max_input_len)

        padding_mask = (torch.arange(max_input_len)[None, :].cpu() >= input_seq_len[:, None].cpu()).to(
            input_seq.device
        )

        memory["encoding_mask"] = padding_mask

        if self.pretrained_encoder is None:
            input_toks_embedded = self.embeddings(input_seq).to(input_seq.device)

            if self.encoder_config.embedding_dim != self.global_config.get_first(
                ["input_raw_embedding_dim", "raw_embedding_dim"]
            ):
                input_toks_embedded = self.embedding_projection(input_toks_embedded)

            input_embedded = input_toks_embedded * math.sqrt(self.encoder_config.embedding_dim)

            memory["seq_embedded"] = input_embedded.detach()

            if include_position:
                input_embedded = self.positional_embeddings(input_embedded)

            memory["seq_embedded_positioned"] = input_embedded.detach()

            encoding = self.encoder(
                input_embedded,
                # is_causal=is_causal,
                # mask=(None if is_causal else src_mask),
                mask=src_mask,
                src_key_padding_mask=padding_mask,
            ).contiguous()

        else:
            # BERT expects a mask that's 1 unmasked, 0 for masked
            bert_padding_mask = (~padding_mask).long()

            bert_typeids: Dict = {}

            bert_encoding = self.pretrained_encoder(
                input_ids=input_seq.to(input_seq.device), attention_mask=bert_padding_mask, **bert_typeids
            )[0]

            if self.encoder_config.get("freeze_pretrained", False):
                bert_encoding = bert_encoding.detach()

            if self.encoder_config.num_layers > 0:

                encoding = self.encoder(
                    bert_encoding,
                    # is_causal=is_causal,
                    # mask=(None if is_causal else src_mask),
                    mask=src_mask,
                    src_key_padding_mask=padding_mask,
                ).contiguous()

            else:
                encoding = bert_encoding

        # Include original input?
        if self.encoder_config.get("residual", False):
            encoding = self.encoder_projection(torch.cat([encoding, input_embedded], dim=-1))

        if self.encoder_config.get("pre_residual", False):
            input_toks_resid = self.embeddings(input_seq.to(self.embeddings.weight.device)).to(input_seq.device)
            input_toks_resid = self.token_projection(input_toks_resid)
            input_toks_resid = self.positional_embeddings(input_toks_resid)
            encoding = torch.cat([encoding, input_toks_resid], dim=-1)

        return encoding, memory
