import math

import torch
import torch.nn as nn
from transformers import BertModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.positional_embeddings import PositionalEncoding
from torchseq.utils.tokenizer import Tokenizer
import torchseq.models.transformer as custom_transformer


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
    return (a + b).unsqueeze(1).clamp(LARGE_NEGATIVE,)


class PretrainedAdapterModel(nn.Module):
    def __init__(self, config, src_field="s1", tgt_field="s1"):
        super().__init__()
        self.config = config

        self.src_field = src_field
        self.tgt_field = tgt_field

        if "mbart" in self.config.encdec.bert_model:
            from transformers import BartModel

            # Encoder/decoders
            bart_model = BartModel.from_pretrained(config.encdec.bert_model)
            self.encoder = bart_model.encoder
            self.decoder = bart_model.decoder

        elif "bart" in self.config.encdec.bert_model:
            from transformers import BartModel

            # Encoder/decoders
            bart_model = BartModel.from_pretrained(config.encdec.bert_model)
            self.encoder = bart_model.encoder
            self.decoder = bart_model.decoder

        self.decoder.generation_mode = False

        if self.config.encdec.data.get("adapter", False):
            if self.config.encdec.get("aq_adapter", False):
                decoder_layer = custom_transformer.TransformerDecoderLayer(
                    config.embedding_dim,
                    nhead=config.encdec.num_heads,
                    dim_feedforward=config.encdec.dim_feedforward,
                    dropout=config.dropout,
                    activation=config.encdec.activation,
                )

                self.adapter = custom_transformer.TransformerDecoder(
                    decoder_layer, config.encdec.num_encoder_layers, None
                )
            else:
                encoder_layer = custom_transformer.TransformerEncoderLayer(
                    config.embedding_dim,
                    nhead=config.encdec.num_heads,
                    dim_feedforward=config.encdec.dim_feedforward,
                    dropout=config.dropout,
                    activation=config.encdec.activation,
                )

                self.adapter = custom_transformer.TransformerEncoder(
                    encoder_layer, config.encdec.num_encoder_layers, None
                )

            for p in self.adapter.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=self.config.encdec.get("adapter_init_scale", 1e-1))

        self.output_projection = nn.Linear(config.embedding_dim, config.prepro.vocab_size, bias=False)
        if config.embedding_dim == config.raw_embedding_dim:
            self.output_projection.weight.data = bart_model.shared.weight.data

        # self.output_projection = bart_model.lm_head
        self.output_projection.weight.requires_grad = not config.freeze_projection

        if config.encdec.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if config.encdec.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, batch, output, memory=None, tgt_field=None):
        if memory is None:
            memory = {}

        # Get some sizes
        max_ctxt_len = batch[self.src_field].shape[1]
        max_goal_len = batch[self.tgt_field].shape[1]
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

            if self.config.encdec.freeze_encoder:
                pretrained_encoding = pretrained_encoding.detach()

            if self.config.encdec.data.get("adapter", False):
                if self.config.encdec.get("aq_adapter", False):
                    sideinfo_mask = batch["a_pos"] == 0

                    sideinfo = pretrained_encoding

                    encoding = self.adapter(
                        pretrained_encoding.transpose(0, 1),
                        sideinfo.permute(1, 0, 2),
                        tgt_key_padding_mask=context_mask,
                        memory_key_padding_mask=sideinfo_mask,
                    ).permute(1, 0, 2)
                    # encoding = self.adapter(pretrained_encoding.transpose(0, 1)).transpose(0, 1)
                else:
                    encoding = self.adapter(
                        pretrained_encoding.transpose(0, 1), src_key_padding_mask=context_mask
                    ).transpose(0, 1)
            else:
                encoding = pretrained_encoding

            memory["encoding"] = encoding

            if self.config.encdec.data.get("adapter", False) and self.config.training.get("mse_loss_weight", 0) > 0:

                # calculate loss
                goal_pad_mask = (
                    torch.arange(max(max_ctxt_len, max_goal_len))[None, :].cpu()
                    >= batch[self.tgt_field + "_len"][:, None].cpu()
                ).to(self.device)
                # TODO: this will fail if the input is shorter than the tgt! eg for paraphrasing
                if max_ctxt_len > max_goal_len:
                    q_padding = torch.full(
                        [batch[self.tgt_field].shape[0], max(max_ctxt_len - max_goal_len, 0)],
                        Tokenizer().pad_id,
                        dtype=torch.long,
                        device=batch[self.tgt_field].device,
                    )
                    goal_padded = torch.cat([batch[self.tgt_field], q_padding], dim=-1)
                else:
                    goal_padded = batch[self.tgt_field]

                goal_encoding = self.encoder(input_ids=goal_padded.to(self.device), attention_mask=~goal_pad_mask)[0]

                if max_ctxt_len < max_goal_len:
                    goal_encoding = goal_encoding[:, :max_ctxt_len, :]

                this_mse_loss = self.mse_loss(encoding, goal_encoding).sum(dim=2) * (1.0 * ~goal_pad_mask)
                this_mse_loss = this_mse_loss.sum(dim=1) / (batch[self.tgt_field + "_len"] - 1).to(encoding)

                if "loss" not in memory:
                    memory["loss"] = 0
                memory["loss"] += this_mse_loss * self.config.training.get("mse_loss_weight", 0)

        # Build some masks
        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float("-1e8")).to(self.device)
        # tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float('0')).to(self.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)

        # ie how many indices are non-pad
        output_len = torch.sum(torch.ne(output, Tokenizer().pad_id), dim=-1)

        output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() >= output_len[:, None].cpu()).to(self.device)[
            :, :output_max_len
        ]

        output = self.decoder(
            output,
            memory["encoding"],
            ~context_mask,
            output_pad_mask,
            decoder_causal_mask=tgt_mask,
            # tgt_key_padding_mask=output_pad_mask
        )

        logits = self.output_projection(output[0])

        return logits, memory
