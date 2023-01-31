import math

import torch
import torch.nn as nn

from torchseq.models.positional_embeddings import PositionalEncoding
from torchseq.models.multihead_output import MultiHeadOutput
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.functions import initialize_truncated_normal_, init_bert_params
from transformers import BartModel, BertModel, RobertaModel, MBartModel


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
            if self.tokenizer.has_embeddings and self.config.encoder.get("init_embeds_from_tokenizer", True):
                self.embeddings.weight.data = self.tokenizer.get_embeddings()
            else:

                if self.config.decoder.get("init_embeds_like_bert", False):
                    init_bert_params(self.embeddings)
                else:
                    torch.nn.init.xavier_uniform_(self.embeddings.weight.data, gain=1.0)
                    # initialize_truncated_normal_(
                    #     self.embeddings.weight.data, std=1 / math.sqrt(config.decoder.embedding_dim)
                    # )
            self.embeddings.weight.requires_grad = not (
                config.decoder.get("freeze_embeddings", config.get("freeze_embeddings", False))
                if "decoder" in config.data
                else config.get("freeze_embeddings", False)
            )
            self.embeddings.cpu()
            self.embeddings.force_device = True

        self.pretrained_model_slug = None
        if config.decoder.get("pretrained_decoder", None) is not None:
            self.pretrained_model_slug = config.decoder.pretrained_decoder
            if "mbart" in self.pretrained_model_slug:
                full_pretrained_model = MBartModel.from_pretrained(self.pretrained_model_slug)
                self.pretrained_decoder = full_pretrained_model.decoder
                # del bart_model
            elif "bart" in self.pretrained_model_slug:
                full_pretrained_model = BartModel.from_pretrained(self.pretrained_model_slug)
                self.pretrained_decoder = full_pretrained_model.decoder
                # del bart_model
            else:
                # TODO: Make this an AutoModel?
                raise Exception("Unrecognised pretrained decoder slug: {:}".format(self.pretrained_model_slug))

            if config.decoder.get("freeze_pretrained", False):
                self.pretrained_decoder.requires_grad = False

            if self.config.decoder.num_layers > 0:
                self.logger.warning("Using non-zero decoder layers with a pretrained decoder - are you sure?")
        else:
            self.pretrained_decoder = None

        decoder_layer = nn.TransformerDecoderLayer(
            config.decoder.embedding_dim,
            nhead=config.decoder.num_heads,
            dim_feedforward=config.decoder.dim_feedforward,
            dropout=config.dropout,
            activation=config.decoder.activation,
            batch_first=True,
        )
        decoder_norm = nn.LayerNorm(config.decoder.embedding_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder.num_layers, decoder_norm)

        if self.config.decoder.get("init_like_bert", False):
            init_bert_params(self.decoder)

        projection_init = None
        if self.pretrained_model_slug is not None:
            # TODO: reuse the model from earlier
            # bart_model = MBartModel.from_pretrained(self.pretrained_model_slug)
            projection_init = full_pretrained_model.shared.weight.data
            del full_pretrained_model
        elif (
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
            if projection_init is not None and self.config.decoder.get("init_embeds_from_tokenizer", True):
                self.output_projection.weight.data = projection_init
            else:
                if self.config.decoder.get("init_embeds_like_bert", False):
                    init_bert_params(self.output_projection)
                else:
                    torch.nn.init.xavier_uniform_(self.output_projection.weight, gain=1.0)
                    # initialize_truncated_normal_(
                    #     self.output_projection.weight, std=1 / math.sqrt(config.decoder.embedding_dim)
                    # )
            self.output_projection.weight.requires_grad = not config.freeze_projection

        self.positional_embeddings = PositionalEncoding(config.decoder.embedding_dim)

    def forward(self, output_seq, memory):

        output_max_len = output_seq.size()[-1]

        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float("-inf")).to(output_seq.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)

        if self.config.decoder.data.get("attention_limit", None) is not None:
            tgt_mask = torch.tril(tgt_mask, diagonal=self.config.decoder.data.get("attention_limit", 0))

        # ie how many indices are non-pad
        output_len = torch.sum(torch.ne(output_seq, self.tokenizer.pad_id), dim=-1)

        output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() >= output_len[:, None].cpu()).to(
            output_seq.device
        )[:, :output_max_len]

        if self.pretrained_model_slug is None:
            # Embed the output so far
            output_embedded = self.embeddings(output_seq).to(output_seq.device) * math.sqrt(
                self.config.decoder.embedding_dim
            )

            # if self.config.raw_embedding_dim != self.config.decoder.embedding_dim:
            #     output_embedded = self.embedding_projection(output_embedded)

            # For some reason the Transformer implementation expects seq x batch x feat - this is weird, so permute the input and the output
            output_embedded = self.positional_embeddings(output_embedded)

            # Decoder block fwd pass
            output_seq = self.decoder(
                output_embedded,
                memory["encoding"],
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=output_pad_mask,
                memory_key_padding_mask=memory["encoding_mask"],
            )

        else:

            # Build some masks
            tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float("-1e8")).to(output_seq.device)
            # tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float('0')).to(self.device)
            tgt_mask = torch.triu(tgt_mask, diagonal=1)

            # ie how many indices are non-pad
            output_len = torch.sum(torch.ne(output_seq, self.tokenizer.pad_id), dim=-1)

            output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() >= output_len[:, None].cpu()).to(
                output_seq.device
            )[:, :output_max_len]

            output_seq = self.pretrained_decoder(
                input_ids=output_seq,
                encoder_hidden_states=memory["encoding"],
                encoder_attention_mask=~memory["encoding_mask"] if memory["encoding_mask"] is not None else None,
                attention_mask=~output_pad_mask,
            )[0]

            if self.config.decoder.num_layers > 0:
                output_seq = self.decoder(
                    output_seq,
                    memory["encoding"],
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=output_pad_mask,
                    memory_key_padding_mask=memory["encoding_mask"],
                )

        # Embeddings -> logits
        if self.config.get("variational_projection", False):
            logits, memory = self.output_projection(output_seq)
        else:
            logits = self.output_projection(output_seq)

        return logits, memory
