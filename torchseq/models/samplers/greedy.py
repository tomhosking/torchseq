from typing import Dict, Union, Tuple
import torch
import torch.nn as nn

from torchseq.utils.tokenizer import Tokenizer, FAIRSEQ_LANGUAGE_CODES
from torchseq.utils.config import Config


class GreedySampler(nn.Module):
    config: Config
    device: Union[str, torch.device]
    tokenizer: Tokenizer

    def __init__(self, config: Config, tokenizer: Tokenizer, device: Union[str, torch.device]):
        super(GreedySampler, self).__init__()
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

    def forward(
        self, model: nn.Module, batch: Dict[str, torch.Tensor], tgt_field: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        curr_batch_size = batch[[k for k in batch.keys() if k[-5:] != "_text"][0]].size()[0]

        max_output_len = self.config.eval.data.get("max_out_len", 32)

        BART_HACK = self.config.eval.data.get("prepend_eos", False)
        MBART_HACK = self.config.eval.data.get("prepend_langcode", False)

        # Create vector of SOS + placeholder for first prediction
        output = torch.LongTensor(curr_batch_size, 1).fill_(self.tokenizer.bos_id).to(self.device)
        logits = (
            torch.FloatTensor(curr_batch_size, 1, self.config.prepro.get_first(["output_vocab_size", "vocab_size"]))
            .fill_(float("-inf"))
            .to(self.device)
        )
        logits[:, :, self.tokenizer.bos_id] = float("inf")

        output_done = torch.BoolTensor(curr_batch_size).fill_(False).to(self.device)
        padding = torch.LongTensor(curr_batch_size).fill_(self.tokenizer.pad_id).to(self.device)

        if BART_HACK:
            dummy_token = torch.LongTensor(curr_batch_size, 1).fill_(self.tokenizer.eos_id).to(self.device)
            output = torch.cat([dummy_token, output], dim=1)

        if MBART_HACK:
            lang_token = batch["tgt_lang"].unsqueeze(-1)
            eos_token = torch.LongTensor(curr_batch_size, 1).fill_(self.tokenizer.eos_id).to(self.device)
            output = torch.cat([eos_token, lang_token], dim=-1)

        seq_ix = 0
        memory: Dict[str, torch.Tensor] = {}
        while torch.sum(output_done) < curr_batch_size and seq_ix < max_output_len:

            new_logits, memory = model(batch, output, memory)

            new_output = torch.argmax(new_logits, -1)

            # Use pad for the output for elements that have completed
            new_output[:, -1] = torch.where(output_done, padding, new_output[:, -1])

            output = torch.cat([output, new_output[:, -1].unsqueeze(-1)], dim=-1)

            logits = torch.cat([logits, new_logits[:, -1:, :]], dim=1)

            output_done = output_done | (output[:, -1] == self.tokenizer.eos_id)
            seq_ix += 1

        # print(BART_HACK, MBART_HACK)
        # print(batch['c'][0])
        # print(batch['q'][0])
        # print(output[0])
        # print(self.tokenizer.decode(output[0]))
        # exit()

        if BART_HACK:
            output = output[:, 1:]
        # if MBART_HACK:
        #     output = output[:, 2:]

        return output, logits, torch.sum(output != self.tokenizer.pad_id, dim=-1), memory
