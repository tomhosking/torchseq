import math
import torch
import torch.nn as nn

from transformers import BertModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.positional_embeddings import PositionalEncoding
import torchseq.models.transformer as custom_transformer
from torchseq.models.lang_predict_loss import LangPredictLoss
from torchseq.utils.functions import evaluating


# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))


LARGE_NEGATIVE = -1e8


def combine_masks(key_padding_mask, causal_lm_mask, targ_size):
    # targ_size = (bsz, tgt_len, src_len)
    a = torch.zeros(targ_size)
    b = torch.zeros(targ_size)
    if key_padding_mask is not None:  # (bsz, tgt_len) -> targ_size
        _check_shapes(key_padding_mask.shape, targ_size[:2])
        reshaped = key_padding_mask.unsqueeze(2).expand(*targ_size)
        a[reshaped] = LARGE_NEGATIVE

    if causal_lm_mask is not None:  # (tgt_len, src_len) -> targ_size
        _check_shapes(causal_lm_mask.shape, targ_size[-2:])
        b = causal_lm_mask.cpu().unsqueeze(0).expand(*targ_size)
    return (
        (a + b)
        .unsqueeze(1)
        .clamp(
            LARGE_NEGATIVE,
        )
    )


class PretrainedAdapterModel(nn.Module):
    def __init__(self, config, input_tokenizer, output_tokenizer, src_field="source", tgt_field="source"):
        super().__init__()
        raise Exception("PretrainedAdapterModel is fully deprecated!")
        self.config = config
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        self.src_field = src_field
        self.tgt_field = tgt_field

        if "mbart" in self.config.encoder.bert_model:
            from transformers import MBartModel

            # Encoder/decoders
            bart_model = MBartModel.from_pretrained(config.encoder.bert_model)
            self.encoder = bart_model.encoder
            self.decoder = bart_model.decoder

        elif "bart" in self.config.encoder.bert_model:
            from transformers import BartModel

            # Encoder/decoders
            bart_model = BartModel.from_pretrained(config.encoder.bert_model)
            self.encoder = bart_model.encoder
            self.decoder = bart_model.decoder

        self.decoder.generation_mode = False

        if self.config.encoder.data.get("adapter", False):
            if self.config.encoder.get("aq_adapter", False):
                decoder_layer = custom_transformer.TransformerDecoderLayer(
                    config.decoder.embedding_dim,
                    nhead=config.encoder.num_heads,
                    dim_feedforward=config.encoder.dim_feedforward,
                    dropout=config.dropout,
                    activation=config.encoder.activation,
                )

                self.adapter = custom_transformer.TransformerDecoder(
                    decoder_layer, config.encoder.num_encoder_layers, None
                )
            else:
                encoder_layer = custom_transformer.TransformerEncoderLayer(
                    config.encoder.embedding_dim,
                    nhead=config.encoder.num_heads,
                    dim_feedforward=config.encoder.dim_feedforward,
                    dropout=config.dropout,
                    activation=config.encoder.activation,
                )

                self.adapter = custom_transformer.TransformerEncoder(
                    encoder_layer, config.encoder.num_encoder_layers, None
                )

            for p in self.adapter.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=self.config.encoder.get("adapter_init_scale", 1e-1))

        self.output_projection = nn.Linear(
            config.decoder.embedding_dim, config.prepro.get_first(["output_vocab_size", "vocab_size"]), bias=False
        )
        if config.decoder.embedding_dim == config.raw_embedding_dim:
            self.output_projection.weight.data = bart_model.shared.weight.data

        # self.output_projection = bart_model.lm_head
        self.output_projection.weight.requires_grad = not config.freeze_projection

        if config.encoder.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if config.decoder.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss(reduction="none")

        if self.config.training.get("lang_loss_weight", 0) > 0:
            self.lang_pred_loss = LangPredictLoss(config)

    def forward(self, batch, output, memory=None, tgt_field=None):
        if memory is None:
            memory = {}

        # Get some sizes
        max_ctxt_len = batch[self.src_field].shape[1]
        # max_goal_len = batch[self.tgt_field].shape[1]
        # max_q_len = torch.max(batch['q_len'])
        # curr_batch_size = batch[self.src_field].size()[0]
        output_max_len = output.size()[-1]

        context_mask = (torch.arange(max_ctxt_len)[None, :].cpu() >= batch[self.src_field + "_len"][:, None].cpu()).to(
            self.device
        )

        # bert_context_mask = ~context_mask

        # bert_context_mask = (1.0 - bert_context_mask.long()) * -10000.0

        if "_test" not in memory:
            memory["_test"] = 0
        else:
            memory["_test"] += 1

        if "loss" not in memory:
            memory["loss"] = 0

        # if memory['_test'] == 2:
        #     print(self.src_field , '->', self.tgt_field)
        #     print(batch['s1'])
        #     print(batch['s2'])
        #     print(output)
        #     # print(context_mask)
        #     exit()

        # First pass? Construct the encoding
        if "encoding" not in memory:

            pretrained_encoding = self.encoder(
                input_ids=batch[self.src_field].to(self.device), attention_mask=~context_mask
            )[0]

            if self.config.encoder.freeze_encoder:
                pretrained_encoding = pretrained_encoding.detach()

            if self.config.encoder.get("adapter", False):
                if self.config.encoder.get("aq_adapter", False):
                    sideinfo_mask = batch["a_pos"] == 0

                    sideinfo = pretrained_encoding

                    encoding = self.adapter(
                        pretrained_encoding.transpose(0, 1),
                        sideinfo,
                        tgt_key_padding_mask=context_mask,
                        memory_key_padding_mask=sideinfo_mask,
                    )
                    # encoding = self.adapter(pretrained_encoding.transpose(0, 1)).transpose(0, 1)
                else:
                    adapter_out = self.adapter(
                        pretrained_encoding.transpose(0, 1), src_key_padding_mask=context_mask
                    ).transpose(0, 1)

                    encoding = torch.cat([pretrained_encoding[:, :1, :], adapter_out[:, 1:, :]], dim=1)
            else:
                encoding = pretrained_encoding

            memory["encoding"] = encoding

            if self.config.training.get("lang_loss_weight", 0) > 0:
                lang_loss_pos = self.lang_pred_loss(encoding.detach(), memory, batch["src_lang"])
                with evaluating(self.lang_pred_loss):
                    lang_loss_neg = self.lang_pred_loss(encoding, memory, batch["src_lang"])

                memory["loss"] += lang_loss_pos + self.config.training.lang_loss_weight * lang_loss_neg

        # Build some masks
        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float("-1e8")).to(self.device)
        # tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float('0')).to(self.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)

        # ie how many indices are non-pad
        output_len = torch.sum(torch.ne(output, self.output_tokenizer.pad_id), dim=-1)

        output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() >= output_len[:, None].cpu()).to(self.device)[
            :, :output_max_len
        ]

        # decoder_attn_mask = combine_masks(output_pad_mask, tgt_mask, (curr_batch_size, output_max_len, output_max_len))

        # print(batch['c'][0])
        # print(batch['q'][0])
        # exit()
        output = self.decoder(
            input_ids=output,
            encoder_hidden_states=memory["encoding"],
            encoder_attention_mask=~context_mask,
            attention_mask=~output_pad_mask,
            # tgt_key_padding_mask=output_pad_mask
        )

        logits = self.output_projection(output[0])

        return logits, memory
