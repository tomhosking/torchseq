
import torch
from torch import nn

import torch.nn.functional as F


from agents.model_agent import ModelAgent

from models.para_transformer import TransformerParaphraseModel
from models.pretrained_modular import PretrainedModularModel
from models.suppression_loss import SuppressionLoss

from datasets.paraphrase_loader import ParaphraseDataLoader
from datasets.squad_loader import SquadDataLoader
from datasets.paraphrase_dataset import ParaphraseDataset
from datasets.squad_dataset import SquadDataset
from utils.tokenizer import BPE

from datasets.paraphrase_pair import ParaphrasePair

class ParaphraseAgent(ModelAgent):

    def __init__(self, config, run_id, silent=False):
        super().__init__(config, run_id, silent)

        
        self.tgt_field = 's1' if self.config.training.data.get('flip_pairs', False) else 's2'

        # define data_loader
        if self.config.training.use_preprocessed_data:
            self.data_loader = PreprocessedDataLoader(config=config)
        else:
            if self.config.training.dataset in ['paranmt', 'parabank', 'kaggle', 'parabank-qs', 'para-squad'] or self.config.training.dataset[:5] == 'qdmr-' or 'kaggle-' in self.config.training.dataset:
                self.data_loader = ParaphraseDataLoader(config=config)
                self.src_field = 's2' if (self.config.task == 'autoencoder' or self.config.training.data.get('flip_pairs', False))  else 's1'
            elif self.config.training.dataset in ['squad']:
                self.data_loader = SquadDataLoader(config=config)
                self.src_field = 'q'
                self.tgt_field = 'q'
            else:
                raise Exception("Unrecognised dataset: {:}".format(config.training.dataset))

        
        # define loss
        self.loss = nn.CrossEntropyLoss(ignore_index=BPE.pad_id, reduction='none')

        # define model
        if self.config.data.get('model', None) is not None and self.config.model == 'pretrained_modular':
            self.model = PretrainedModularModel(self.config, src_field=self.src_field, loss=self.loss)
        else:
            self.model = TransformerParaphraseModel(self.config, src_field=self.src_field, loss=self.loss)



        self.suppression_loss = SuppressionLoss(self.config)

        # define optimizer
        self.create_optimizer()

        self.set_device()

        self.create_samplers()


    def step_train(self, batch, tgt_field):
        loss = 0
            
        output, logits, this_loss = self.decode_teacher_force(self.model, batch, tgt_field)

        # this_loss = self.loss(logits.permute(0,2,1), batch[tgt_field])

        if self.config.training.suppression_loss_weight > 0:
            this_loss +=  self.config.training.suppression_loss_weight * self.suppression_loss(logits, batch['s1'])

        this_loss = torch.sum(this_loss, dim=1)/batch[tgt_field+'_len'].to(this_loss)

        loss_weight = torch.where(batch['is_paraphrase'] > 0, torch.full_like(this_loss, 1.0), torch.full_like(this_loss, -1.0 * self.config.training.data.get('neg_sample_weight', 0)))
        
        loss += torch.mean(loss_weight * this_loss, dim=0)

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
        if self.config.training.dataset in ['squad']:
            x['s2'] = ''
            
            return {k: (v.to(self.device) if k[-5:] != '_text' else v) for k,v in SquadDataset.pad_and_order_sequences([SquadDataset.to_tensor(x, tok_window=self.config.prepro.tok_window)]).items()}
        else:
            x['s2'] = ''
            
            return {k: (v.to(self.device) if k[-5:] != '_text' else v) for k,v in ParaphraseDataset.pad_and_order_sequences([ParaphraseDataset.to_tensor(x, tok_window=self.config.prepro.tok_window)]).items()}

        