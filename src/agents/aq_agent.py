import numpy as np

from tqdm import tqdm
import shutil
import random
import json

from args import FLAGS as FLAGS


import torch
from torch import nn

from torch.autograd import Variable

import torch.nn.functional as F

from agents.model_agent import ModelAgent

from models.aq_transformer import TransformerAqModel

from datasets.squad_loader import SquadDataLoader
from datasets.newsqa_loader import NewsqaDataLoader
from datasets.preprocessed_loader import PreprocessedDataLoader
from datasets.loaders import load_glove, get_embeddings

from utils.logging import add_to_log

from utils.misc import print_cuda_statistics
from utils.metrics import bleu_corpus

from utils.bpe_factory import BPE
from utils.mckenzie import update_mckenzie

from models.suppression_loss import SuppressionLoss


from models.cross_entropy import CrossEntropyLossWithLS

import os




class AQAgent(ModelAgent):

    def __init__(self, config, run_id, silent=False):
        super().__init__(config, run_id, silent)

        self.tgt_field = 'q'

        # define models
        self.model = TransformerAqModel(config)

        # define data_loader
        if self.config.training.use_preprocessed_data:
            self.data_loader = PreprocessedDataLoader(config=config)
        else:
            if self.config.training.dataset == 'squad':
                self.data_loader = SquadDataLoader(config=config)
            elif self.config.training.dataset == 'newsqa':
                self.data_loader = NewsqaDataLoader(config=config)
            else:
                raise Exception("Unrecognised dataset: {:}".format(config.training.dataset))

        # define loss
        if self.config.training.label_smoothing != "UNUSED" and self.config.training.label_smoothing > 1e-6:
            self.loss = CrossEntropyLossWithLS(ignore_index=BPE.pad_id, smooth_eps=self.config.training.label_smoothing, reduction='none' )
        else:
            self.loss = nn.CrossEntropyLoss(ignore_index=BPE.pad_id, reduction='none')

        self.suppression_loss = SuppressionLoss(self.config)

        # define optimizer
        self.create_optimizer()

        self.set_device()

        if self.cuda:
            self.suppression_loss = self.suppression_loss.to(self.device)

        self.create_samplers()

    def begin_epoch_hook(self):
        self.model.freeze_bert = self.current_epoch >= self.config.encdec.bert_warmup_epochs


    def step_train(self, batch, tgt_field):
        loss = 0
            
        output, logits = self.decode_teacher_force(self.model, batch, 'q')

        this_loss = self.loss(logits.permute(0,2,1), batch['q'])

        if self.config.training.suppression_loss_weight > 0:
            this_loss +=  self.config.training.suppression_loss_weight * self.suppression_loss(logits, batch['a'])
        
        loss += torch.mean(torch.sum(this_loss, dim=1)/batch['q_len'].to(this_loss), dim=0)

        return loss


    

        


