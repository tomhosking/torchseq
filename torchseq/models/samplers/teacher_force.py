from typing import Dict, Union, Tuple
import torch
import torch.nn as nn

from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.config import Config


class TeacherForcedSampler(nn.Module):
    config: Config
    device: Union[str, torch.device]
    tokenizer: Tokenizer

    def __init__(self, config: Config, tokenizer: Tokenizer, device: Union[str, torch.device]):
        super(TeacherForcedSampler, self).__init__()
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

    def forward(
        self, model: nn.Module, batch: Dict[str, torch.Tensor], tgt_field: str
    ) -> Tuple[torch.Tensor, torch.Tensor, None, Dict[str, torch.Tensor]]:
        curr_batch_size = batch[[k for k in batch.keys() if k[-5:] != "_text"][0]].size()[0]
        max_output_len = batch[tgt_field].size()[1]

        BART_HACK = self.config.eval.data.get("prepend_eos", False)
        MBART_HACK = self.config.eval.data.get("prepend_langcode", False)

        # Create vector of SOS + placeholder for first prediction

        logits = (
            torch.FloatTensor(curr_batch_size, 1, self.config.prepro.get_first(["output_vocab_size", "vocab_size"]))
            .fill_(float("-1e18"))
            .to(self.device)
        )

        if MBART_HACK:
            logits.scatter_(-1, batch["tgt_lang"].unsqueeze(1), float("1e18"))
        else:
            logits[:, :, self.tokenizer.bos_id] = float("1e18")

        # With a transformer decoder, we can lean on the internal mask to ensure that the model can't see ahead
        # ..and then just do a single pass through the whole model using the gold output as input
        output = batch[tgt_field][:, : max_output_len - 1].to(self.device)

        if self.config.training.data.get("token_dropout", 0) > 0 and self.training:
            rand = torch.rand_like(output, dtype=torch.float)

            masked = torch.full_like(output, self.tokenizer.mask_id)

            output = torch.where(
                torch.bitwise_and(
                    rand < self.config.training.data.get("token_dropout", 0), output != self.tokenizer.pad_id
                ),
                masked,
                output,
            )

        if BART_HACK:
            dummy_token = torch.LongTensor(curr_batch_size, 1).fill_(self.tokenizer.eos_id).to(self.device)
            output = torch.cat([dummy_token, output], dim=1)
        if MBART_HACK:
            eos_token = torch.LongTensor(curr_batch_size, 1).fill_(self.tokenizer.eos_id).to(self.device)

            # lang_token = batch["tgt_lang"].unsqueeze(-1)

            output = torch.cat([eos_token, output], dim=1)
            # print(output[0])
            # exit()

        memory: Dict[str, torch.Tensor] = {}
        pred_logits, memory = model(batch, output, tgt_field=tgt_field, memory=memory)

        if BART_HACK or MBART_HACK:
            output = output[:, 1:]

            pred_logits = pred_logits[:, 1:, :]
        # if MBART_HACK:
        #     output = output[:, 2:]

        #     pred_logits = pred_logits[:, 2:, :]

        logits = torch.cat([logits, pred_logits], dim=1)

        # print(BART_HACK, MBART_HACK)
        # print(output)
        # print(batch['q'])
        # print(torch.argmax(logits, dim=-1))
        # exit()

        return output, logits, None, memory
