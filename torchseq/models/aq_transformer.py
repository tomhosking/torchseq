import math

import torch
import torch.nn as nn

# from transformers import BartModel, BertModel, RobertaModel

# from torchseq.models.pooling import MultiHeadedPooling
# from torchseq.models.positional_embeddings import PositionalEncoding
# from torchseq.models.multihead_output import MultiHeadOutput
# from torchseq.utils.tokenizer import Tokenizer


# import torchseq.models.transformer as custom_transformer

from torchseq.models.encoder import ContextAnswerEncoder
from torchseq.models.decoder import SequenceDecoder


class TransformerAqModel(nn.Module):
    def __init__(self, config, loss=None):
        super().__init__()
        self.config = config

        self.loss = loss

        self.ctxt_ans_encoder = ContextAnswerEncoder(config)
        self.seq_decoder = SequenceDecoder(config, embeddings=self.ctxt_ans_encoder.embeddings)

        # # Embedding layers
        # self.embeddings = nn.Embedding(config.prepro.vocab_size, config.raw_embedding_dim).cpu()
        # self.embeddings.weight.data = Tokenizer().get_embeddings(config.prepro.tokenizer)
        # self.embeddings.weight.requires_grad = not config.freeze_embeddings

        # self.embedding_projection = nn.utils.weight_norm(
        #     nn.Linear(config.raw_embedding_dim, config.embedding_dim, bias=False)
        # )
        # self.bert_embedding_projection = nn.utils.weight_norm(
        #     nn.Linear(config.embedding_dim * 1 + config.bio_embedding_dim, config.embedding_dim, bias=False)
        # )

        # self.bio_embeddings = (
        #     nn.Embedding.from_pretrained(torch.eye(config.bio_embedding_dim), freeze=True).cpu()
        #     if config.onehot_bio
        #     else nn.Embedding(3, config.bio_embedding_dim).cpu()
        # )

        # # Encoder/decoders
        # if config.encdec.bert_encoder:
        #     self.freeze_bert = not self.config.encdec.bert_finetune
        #     # if not self.config.encdec.bert_finetune:
        #     #     print("Building bert encoder without grads")
        #     #     with torch.no_grad():
        #     #         self.bert_encoder = BertModel.from_pretrained(config.encdec.bert_model)

        #     #     for param in self.bert_encoder.parameters():
        #     #         param.requires_grad = False
        #     # else:
        #     if "bart-" in config.encdec.bert_model:
        #         bart_model = BartModel.from_pretrained(config.encdec.bert_model)
        #         self.bert_encoder = bart_model.encoder
        #         del bart_model.decoder
        #     elif "roberta-" in config.encdec.bert_model:
        #         self.bert_encoder = RobertaModel.from_pretrained(config.encdec.bert_model)
        #     else:
        #         self.bert_encoder = BertModel.from_pretrained(config.encdec.bert_model)
        #     # self.bert_encoder.train()
        #     # for param in self.bert_encoder.parameters():
        #     #     param.requires_grad = True

        # if self.config.encdec.data.get("pre_ln", False):
        #     encoder_layer = custom_transformer.TransformerEncoderLayer(
        #         config.embedding_dim + (0 if config.encdec.bert_encoder else config.bio_embedding_dim),
        #         nhead=config.encdec.num_heads,
        #         dim_feedforward=config.encdec.dim_feedforward,
        #         dropout=config.dropout,
        #         activation=config.encdec.activation,
        #     )
        #     encoder_norm = nn.LayerNorm(
        #         config.embedding_dim + (0 if config.encdec.bert_encoder else config.bio_embedding_dim)
        #     )
        #     self.encoder = custom_transformer.TransformerEncoder(
        #         encoder_layer, config.encdec.num_encoder_layers, encoder_norm
        #     )

        #     decoder_layer = custom_transformer.TransformerDecoderLayer(
        #         config.embedding_dim,
        #         nhead=config.encdec.num_heads,
        #         dim_feedforward=config.encdec.dim_feedforward,
        #         dropout=config.dropout,
        #         activation=config.encdec.activation,
        #     )
        #     decoder_norm = nn.LayerNorm(config.embedding_dim)
        #     self.decoder = custom_transformer.TransformerDecoder(
        #         decoder_layer, config.encdec.num_decoder_layers, decoder_norm
        #     )
        # else:
        #     encoder_layer = nn.TransformerEncoderLayer(
        #         config.embedding_dim + (0 if config.encdec.bert_encoder else config.bio_embedding_dim),
        #         nhead=config.encdec.num_heads,
        #         dim_feedforward=config.encdec.dim_feedforward,
        #         dropout=config.dropout,
        #         activation=config.encdec.activation,
        #     )
        #     encoder_norm = nn.LayerNorm(
        #         config.embedding_dim + (0 if config.encdec.bert_encoder else config.bio_embedding_dim)
        #     )
        #     self.encoder = nn.TransformerEncoder(encoder_layer, config.encdec.num_encoder_layers, encoder_norm)

        #     decoder_layer = nn.TransformerDecoderLayer(
        #         config.embedding_dim,
        #         nhead=config.encdec.num_heads,
        #         dim_feedforward=config.encdec.dim_feedforward,
        #         dropout=config.dropout,
        #         activation=config.encdec.activation,
        #     )
        #     decoder_norm = nn.LayerNorm(config.embedding_dim)
        #     self.decoder = nn.TransformerDecoder(decoder_layer, config.encdec.num_decoder_layers, decoder_norm)

        # if self.config.encdec.data.get("xav_uni_init", False):
        #     for p in self.encoder.parameters():
        #         if p.dim() > 1:
        #             nn.init.xavier_uniform_(p)
        #     for p in self.decoder.parameters():
        #         if p.dim() > 1:
        #             nn.init.xavier_uniform_(p)

        # # Encoder combination
        # num_encoder_outputs = sum(
        #     [1 if v else 0 for k, v in config.encoder_outputs.data.items() if k != "c_ans_labels"]
        # )
        # memory_dim = (
        #     config.embedding_dim + (0 if config.encdec.bert_encoder else config.bio_embedding_dim)
        # ) * num_encoder_outputs
        # memory_dim += self.config.bio_embedding_dim if self.config.encoder_outputs.c_ans_labels else 0
        # self.encoder_projection = nn.utils.weight_norm(nn.Linear(memory_dim, config.embedding_dim, bias=False))

        # # self.output_projection = nn.Linear(config.embedding_dim, config.prepro.vocab_size, bias=False).cpu()

        # # # Force the various projections to be unit norm so they can't block information flow
        # # # with torch.no_grad():
        # # #     self.embedding_projection.weight.div_(torch.norm(self.embedding_projection.weight, dim=1, keepdim=True))
        # # #     self.bert_embedding_projection.weight.div_(torch.norm(self.bert_embedding_projection.weight, dim=1, keepdim=True))
        # # #     self.encoder_projection.weight.div_(torch.norm(self.encoder_projection.weight, dim=1, keepdim=True))

        # # # Init output projection layer with embedding matrix
        # # if config.embedding_dim == config.raw_embedding_dim:
        # #     self.output_projection.weight.data = self.embeddings.weight.data
        # # self.output_projection.weight.requires_grad = not config.freeze_projection

        # if config.embedding_dim == config.raw_embedding_dim and config.data.get("init_projection", True):
        #     projection_init = self.embeddings.weight.data
        # else:
        #     projection_init = None

        # if config.data.get("output_projection_heads", 1) > 1 or self.config.data.get("variational_projection", False):
        #     self.output_projection = MultiHeadOutput(
        #         config.embedding_dim,
        #         config.prepro.vocab_size,
        #         num_heads=config.data.get("output_projection_heads", 1),
        #         projection_init=projection_init,
        #         freeze_projection=config.freeze_projection,
        #         variational=self.config.data.get("variational_projection", False),
        #     )
        # else:
        #     self.output_projection = nn.Linear(config.embedding_dim, config.prepro.vocab_size, bias=False).cpu()
        #     # Init output projection layer with embedding matrix
        #     if projection_init is not None:
        #         self.output_projection.weight.data = projection_init
        #     self.output_projection.weight.requires_grad = not config.freeze_projection

        # # Pooling layers
        # self.ans_pooling = MultiHeadedPooling(
        #     config.encdec.num_heads,
        #     config.embedding_dim + config.bio_embedding_dim,
        #     dropout=config.dropout,
        #     model_dim_out=config.embedding_dim,
        #     use_final_linear=False,
        # )
        # self.ctxt_pooling = MultiHeadedPooling(
        #     config.encdec.num_heads,
        #     config.embedding_dim + config.bio_embedding_dim,
        #     dropout=config.dropout,
        #     model_dim_out=config.embedding_dim,
        #     use_final_linear=False,
        #     use_bilinear=True,
        # )

        # # Position encoding
        # self.positional_embeddings_enc = PositionalEncoding(
        #     config.embedding_dim + (0 if config.encdec.bert_encoder else config.bio_embedding_dim)
        # )
        # self.positional_embeddings_dec = PositionalEncoding(config.embedding_dim)

    def forward(self, batch, output, memory=None, tgt_field=None):
        if memory is None:
            memory = dict()

        if "encoding" not in memory:
            encoding, memory = self.ctxt_ans_encoder(batch["c"], batch["c_len"], batch["a_pos"], memory)
            memory["encoding"] = encoding

        # output_max_len = output.size()[-1]

        # # Build some masks
        # tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float("-inf")).to(self.device)
        # tgt_mask = torch.triu(tgt_mask, diagonal=1)

        # # ie how many indices are non-pad
        # output_len = torch.sum(torch.ne(output, Tokenizer().pad_id), dim=-1)

        # output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() >= output_len[:, None].cpu()).to(self.device)[
        #     :, :output_max_len
        # ]

        # # Embed the output so far, then do a decoder fwd pass
        # output_embedded = self.embeddings(output).to(self.device) * math.sqrt(self.config.embedding_dim)

        # if self.config.raw_embedding_dim != self.config.embedding_dim:
        #     output_embedded = self.embedding_projection(output_embedded)

        # # For some reason the Transformer implementation expects seq x batch x feat - this is weird, so permute the input and the output
        # output_embedded = self.positional_embeddings_dec(output_embedded.permute(1, 0, 2))

        # output = self.decoder(
        #     output_embedded,
        #     memory["encoding"].permute(1, 0, 2),
        #     tgt_mask=tgt_mask,
        #     tgt_key_padding_mask=output_pad_mask,
        #     memory_key_padding_mask=memory["encoding_mask"],
        # ).permute(1, 0, 2)

        # logits = self.output_projection(output)

        logits, memory = self.seq_decoder(output, memory)

        return logits, memory
