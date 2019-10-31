import torch
import torch.nn as nn

from utils.bpe_factory import BPE

class TeacherForcedSampler(nn.Module):
    def __init__(self, config, device):
        super(TeacherForcedSampler, self).__init__()
        self.config = config
        self.device = device

    def forward(self, model, batch):
        curr_batch_size = batch['c'].size()[0]
        max_output_len = batch['q'].size()[1]

        # Create vector of SOS + placeholder for first prediction
        
        
        logits = torch.FloatTensor(curr_batch_size, 1, self.config.prepro.vocab_size+1).fill_(float('-1e18')).to(self.device)
        logits[:, :, BPE.instance().BOS] = float('1e18')

        # With a transformer decoder, we can lean on the internal mask to ensure that the model can't see ahead
        # ..and then just do a single pass through the whole model using the gold output as input
        output = batch['q'][:, :max_output_len-1].to(self.device)
        pred_logits, _ = model(batch, output)

        logits = torch.cat([logits, pred_logits], dim=1)
        

        return output, logits