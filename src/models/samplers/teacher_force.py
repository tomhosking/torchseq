import torch
import torch.nn as nn

from utils.tokenizer import BPE

class TeacherForcedSampler(nn.Module):
    def __init__(self, config, device):
        super(TeacherForcedSampler, self).__init__()
        self.config = config
        self.device = device

    def forward(self, model, batch, tgt_field):
        curr_batch_size = batch[[k for k in batch.keys()][0]].size()[0]
        max_output_len = batch[tgt_field].size()[1]

        BART_HACK = self.config.eval.data.get('prepend_eos', False)

        # Create vector of SOS + placeholder for first prediction

        
        logits = torch.FloatTensor(curr_batch_size, 1, self.config.prepro.vocab_size).fill_(float('-1e18')).to(self.device)
        logits[:, :, BPE.bos_id] = float('1e18')


        # With a transformer decoder, we can lean on the internal mask to ensure that the model can't see ahead
        # ..and then just do a single pass through the whole model using the gold output as input
        output = batch[tgt_field][:, :max_output_len-1].to(self.device)


        if BART_HACK:
            dummy_token = torch.LongTensor(curr_batch_size, 1).fill_(BPE.eos_id).to(self.device)
            output = torch.cat([dummy_token, output], dim=1)

        pred_logits, _, this_loss = model(batch, output, tgt_field=tgt_field)

        

        


        if BART_HACK:
            output = output[:, 1:]
    
            pred_logits = pred_logits[:, 1:, :]

        logits = torch.cat([logits, pred_logits], dim=1)
        

        return output, logits, this_loss