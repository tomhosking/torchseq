import torch
import torch.nn as nn

from torchseq.utils.tokenizer import Tokenizer


class GreedySampler(nn.Module):
    def __init__(self, config, device):
        super(GreedySampler, self).__init__()
        self.config = config
        self.device = device

    def forward(self, model, batch, tgt_field):
        curr_batch_size = batch[[k for k in batch.keys()][0]].size()[0]

        max_output_len = batch[tgt_field].size()[1]

        BART_HACK = self.config.eval.data.get("prepend_eos", False)
        MBART_HACK = self.config.eval.data.get("prepend_langcode", False)

        # Create vector of SOS + placeholder for first prediction
        output = torch.LongTensor(curr_batch_size, 1).fill_(Tokenizer().bos_id).to(self.device)
        logits = (
            torch.FloatTensor(curr_batch_size, 1, self.config.prepro.vocab_size).fill_(float("-inf")).to(self.device)
        )
        logits[:, :, Tokenizer().bos_id] = float("inf")

        output_done = torch.BoolTensor(curr_batch_size).fill_(False).to(self.device)
        padding = torch.LongTensor(curr_batch_size).fill_(Tokenizer().pad_id).to(self.device)

        if BART_HACK:
            dummy_token = torch.LongTensor(curr_batch_size, 1).fill_(Tokenizer().eos_id).to(self.device)
            output = torch.cat([dummy_token, output], dim=1)

        if MBART_HACK:
            dummy_token = torch.LongTensor(curr_batch_size, 1).fill_(250004).to(self.device)
            output = torch.cat([dummy_token, output], dim=1)
            # output = dummy_token
            # scores = torch.cat([scores, scores], dim=-1)

        seq_ix = 0
        memory = {}
        while torch.sum(output_done) < curr_batch_size and seq_ix < max_output_len:

            new_logits, memory = model(batch, output, memory)

            new_output = torch.argmax(new_logits, -1)

            # Use pad for the output for elements that have completed
            new_output[:, -1] = torch.where(output_done, padding, new_output[:, -1])

            output = torch.cat([output, new_output[:, -1].unsqueeze(-1)], dim=-1)

            logits = torch.cat([logits, new_logits[:, -1:, :]], dim=1)

            output_done = output_done | (output[:, -1] == Tokenizer().eos_id)
            seq_ix += 1

        if BART_HACK or MBART_HACK:
            output = output[:, 1:]

        return output, logits, torch.sum(output != Tokenizer().pad_id, dim=-1), memory
