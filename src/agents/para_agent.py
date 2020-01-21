
import torch
from torch import nn

import torch.nn.functional as F


from agents.model_agent import ModelAgent

from models.para_transformer import TransformerParaphraseModel
from models.suppression_loss import SuppressionLoss

from datasets.paraphrase_loader import ParaphraseDataLoader
from datasets.squad_loader import SquadDataLoader
from datasets.paraphrase_dataset import ParaphraseDataset
from utils.bpe_factory import BPE

from datasets.paraphrase_pair import ParaphrasePair

class ParaphraseAgent(ModelAgent):

    def __init__(self, config, run_id, silent=False):
        super().__init__(config, run_id, silent)

        
        self.tgt_field = 's2'

        # define data_loader
        if self.config.training.use_preprocessed_data:
            self.data_loader = PreprocessedDataLoader(config=config)
        else:
            if self.config.training.dataset in ['paranmt', 'parabank', 'kaggle', 'parabank-qs']:
                self.data_loader = ParaphraseDataLoader(config=config)
                self.src_field = 's1'
            elif self.config.training.dataset in ['squad']:
                self.data_loader = SquadDataLoader(config=config)
                self.src_field = 'q'
                self.tgt_field = 'q'
            else:
                raise Exception("Unrecognised dataset: {:}".format(config.training.dataset))

        # define model
        self.model = TransformerParaphraseModel(self.config, src_field=self.src_field)


        # define loss
        self.loss = nn.CrossEntropyLoss(ignore_index=BPE.pad_id, reduction='none')

        self.suppression_loss = SuppressionLoss(self.config)

        # define optimizer
        self.create_optimizer()

        self.set_device()

        self.create_samplers()


    def step_train(self, batch, tgt_field):
        loss = 0
            
        output, logits = self.decode_teacher_force(self.model, batch, tgt_field)

        this_loss = self.loss(logits.permute(0,2,1), batch[tgt_field])

        if self.config.training.suppression_loss_weight > 0:
            this_loss +=  self.config.training.suppression_loss_weight * self.suppression_loss(logits, batch['s1'])
        
        loss += torch.mean(torch.sum(this_loss, dim=1)/batch[tgt_field+'_len'].to(this_loss), dim=0)

        return loss


    # def pad_and_order_sequences(self, batch):
    #     keys = batch[0].keys()
    #     max_lens = {k: max(len(x[k]) for x in batch) for k in keys}

    #     for x in batch:
    #         for k in keys:
    #             if k == 'a_pos':
    #                 x[k] = F.pad(x[k], (0, max_lens[k]-len(x[k])), value=0)
    #             else:
    #                 x[k] = F.pad(x[k], (0, max_lens[k]-len(x[k])), value=BPE.pad_id)

    #     tensor_batch = {}
    #     for k in keys:
    #         tensor_batch[k] = torch.stack([x[k] for x in batch], 0).squeeze(dim=1)

    #     return tensor_batch

        
    def text_to_batch(self, x, device):

        x['s2'] = ''

        return {k:v.to(device) for k,v in ParaphraseDataset.pad_and_order_sequences([ParaphraseDataset.to_tensor(x, tok_window=self.config.prepro.tok_window)]).items()}

        # parsed_triple = ParaphrasePair(x['s1'], "", config=self.config)

        # sample = {
        #         's1': torch.LongTensor(parsed_triple.s1_as_ids()).to(self.device),
        #         # 's2': torch.LongTensor(parsed_triple.s2_as_ids()),
        #         's1_len': torch.LongTensor([len(parsed_triple._s1_doc)]).to(self.device),
        #         # 's2_len': torch.LongTensor([len(parsed_triple._s2_doc)])
        #         }
            
                    

        # return self.pad_and_order_sequences([sample])