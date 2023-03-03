import math

import torch
import torch.nn as nn

from torchseq.models.ctxtans_encoder import ContextAnswerEncoder
from torchseq.models.decoder import SequenceDecoder


class TransformerAqModel(nn.Module):
    def __init__(self, config, input_tokenizer, output_tokenizer, loss=None):
        super().__init__()
        self.config = config

        self.loss = loss

        self.ctxt_ans_encoder = ContextAnswerEncoder(config, input_tokenizer)
        self.seq_decoder = SequenceDecoder(config, output_tokenizer, embeddings=self.ctxt_ans_encoder.embeddings)

    def forward(self, batch, output, memory=None, tgt_field=None):
        if memory is None:
            memory = dict()

        if "encoding" not in memory:
            encoding, memory = self.ctxt_ans_encoder(batch["c"], batch["c_len"], batch["a_pos"], memory)
            memory["encoding"] = encoding

        logits, memory = self.seq_decoder(output, memory)

        return logits, memory
