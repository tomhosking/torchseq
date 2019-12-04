
import torch
import torch.nn as nn

from utils.bpe_factory import BPE

class GreedySampler(nn.Module):
    def __init__(self, config, device):
        super(GreedySampler, self).__init__()
        self.config = config
        self.device = device

    def forward(self, model, batch):
        curr_batch_size = batch['c'].size()[0]
        max_output_len = batch['q'].size()[1]

        # Create vector of SOS + placeholder for first prediction
        output = torch.LongTensor(curr_batch_size, 1).fill_(BPE.bos_id).to(self.device)
        logits = torch.FloatTensor(curr_batch_size, 1, self.config.prepro.vocab_size).fill_(float('-inf')).to(self.device)
        logits[:, :, BPE.bos_id] = float('inf')

        output_done = torch.BoolTensor(curr_batch_size).fill_(False).to(self.device)
        padding = torch.LongTensor(curr_batch_size).fill_(BPE.instance().pad_token_id).to(self.device)

        seq_ix = 0
        memory = None
        while torch.sum(output_done) < curr_batch_size and seq_ix < max_output_len:
            new_logits, memory = model(batch, output, memory)

            new_output = torch.argmax(new_logits, -1)

            new_scores = torch.max(new_logits, -1).values
            # print(new_scores)

            # Use pad for the output for elements that have completed
            new_output[:, -1] = torch.where(output_done, padding, new_output[:, -1])
            
            output = torch.cat([output, new_output[:, -1].unsqueeze(-1)], dim=-1)

            logits = torch.cat([logits, new_logits[:, -1:, :]], dim=1)

            # print(output_done)
            # print(output[:, -1])
            # print(output[:, -1] == BPE.eos_id)
            output_done = output_done | (output[:, -1] == BPE.eos_id)
            seq_ix += 1
        # exit()
        
        # output.where(output == BPE.pad_id, torch.LongTensor(output.shape).fill_(-1).to(self.device))

        return output, logits, torch.sum(output != BPE.pad_id, dim=-1)