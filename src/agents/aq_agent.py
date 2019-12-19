import numpy as np

from tqdm import tqdm
import shutil
import random
import json

from args import FLAGS as FLAGS


import torch
from torch import nn
from torch.backends import cudnn
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

from models.suppression_loss import SuppressionLoss

from models.lr_schedule import get_lr

import os

cudnn.benchmark = True


class AQAgent(ModelAgent):

    def __init__(self, config, run_id, silent=False):
        super().__init__(config, run_id, silent)

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
        self.loss = nn.CrossEntropyLoss(ignore_index=BPE.pad_id, reduction='none')
        # self.loss = CrossEntropyLossWithLS(ignore_index=BPE.pad_id, smooth_eps=self.config.training.label_smoothing, )

        self.suppression_loss = SuppressionLoss(self.config)

        # define optimizer
        self.create_optimizer()

        self.set_device()

        self.model.device = self.device

        self.create_samplers()



    def train(self):
        """
        Main training loop
        :return:
        """
        self.global_idx = self.current_epoch * len(self.data_loader.train_loader.dataset)
        for epoch in range(self.config.training.num_epochs):
            self.model.freeze_bert = self.current_epoch >= self.config.encdec.bert_warmup_epochs
            self.train_one_epoch()

            self.current_epoch += 1

            if self.current_epoch > self.config.training.warmup_epochs:
                self.validate(save=True)

        self.logger.info('## Training completed {:} epochs'.format(self.current_epoch+1))
        self.logger.info('## Best metrics: {:}'.format(self.all_metrics_at_best))


    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        self.model.train()

        self.logger.info('## Training epoch {:}'.format(self.current_epoch))
        
        self.optimizer.zero_grad()
        steps_accum = 0
        
        for batch_idx, batch in enumerate(tqdm(self.data_loader.train_loader, desc='Epoch {:}'.format(self.current_epoch), disable=self.silent)):
            batch= {k:v.to(self.device) for k,v in batch.items()}
            curr_batch_size = batch['c'].size()[0]
            max_output_len = batch['q'].size()[1]
            self.global_idx += curr_batch_size
            
            loss = 0
            
            output, logits = self.decode_teacher_force(self.model, batch)

            this_loss = self.loss(logits.permute(0,2,1), batch['q'])

            if self.config.training.suppression_loss_weight > 0:
                this_loss +=  self.config.training.suppression_loss_weight * self.suppression_loss(logits, batch['a'])
            
            loss += torch.mean(torch.sum(this_loss, dim=1)/batch['q_len'].to(this_loss), dim=0)
            
            
            loss.backward()

            steps_accum += curr_batch_size
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_gradient)
            
            lr = get_lr(self.config.training.lr, self.current_iteration, self.config.training.lr_schedule)

            add_to_log('train/lr', lr, self.current_iteration, self.config.tag +'/' + self.run_id)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Gradient accumulation
            if steps_accum >= self.config.training.optim_batch_size:
                self.optimizer.step()
                self.optimizer.zero_grad()
                steps_accum = 0
                self.current_iteration += 1
            
            if batch_idx % self.config.training.log_interval == 0:
                add_to_log('train/loss', loss, self.current_iteration, self.config.tag +'/' + self.run_id)
                
                # # print(batch['q'][0])
                # # print(greedy_output.data[0])
                if batch_idx % (self.config.training.log_interval*20) == 0 and not self.silent:

                    with torch.no_grad():
                        greedy_output, _, output_lens = self.decode_greedy(self.model, batch)
                    
                    print(BPE.decode(batch['q'][0][:batch['q_len'][0]]))
                    print(BPE.decode(greedy_output.data[0][:output_lens[0]]))

                torch.cuda.empty_cache()

    def validate(self, save=False, force_save_output=False, use_test=False):
        """
        One cycle of model validation
        :return:
        """
        self.logger.info('## Validating after {:} epochs'.format(self.current_epoch))
        self.model.eval()
        test_loss = 0
        correct = 0

        q_preds = []
        q_golds = []

        if use_test:
            print('***** USING TEST SET ******')
            valid_loader = self.data_loader.test_loader
        else:
            valid_loader = self.data_loader.valid_loader

        with torch.no_grad():
            num_batches = 0
            for batch_idx, batch in enumerate(tqdm(valid_loader, desc='Validating after {:} epochs'.format(self.current_epoch), disable=self.silent)):
                batch= {k:v.to(self.device) for k,v in batch.items()}
                curr_batch_size = batch['c'].size()[0]
                max_output_len = batch['q'].size()[1]

                

                _, logits = self.decode_teacher_force(self.model, batch)
                
                # dev_output, _, dev_output_lens = self.decode_greedy(self.model, batch)
                beam_output, _, beam_lens = self.decode_beam(self.model, batch)

                # if batch_idx == 0:
                #     print('OUTPUT OF BEAM')
                #     for i in range(beam_output.shape[1]):
                #         print(BPE.decode(beam_output.data[0,i][:output_lens[0,i]]))
                #     exit()

                dev_output = beam_output[:, 0, :]
                dev_output_lens = beam_lens[:, 0]

                this_loss = self.loss(logits.permute(0,2,1), batch['q'])
            
                test_loss += torch.mean(torch.sum(this_loss, dim=1)/batch['q_len'].to(this_loss), dim=0)
                num_batches += 1

                for ix, q_pred in enumerate(dev_output.data):
                    q_preds.append(BPE.decode(q_pred[:dev_output_lens[ix]]))
                for ix, q_gold in enumerate(batch['q']):
                    q_golds.append(BPE.decode(q_gold[:batch['q_len'][ix]]))
                

                if batch_idx % 200 == 0 and not self.silent:
                    # print(batch['c'][0][:10])
                    # for ix in range(batch['c_len'][-2]):
                    #     print(" ".join([BPE.instance().pieces[batch['c'][-2][ix]] +'//' + str(batch['a_pos'][-2][ix])]))
                    print(BPE.decode(batch['c'][-2][:batch['c_len'][-2]]))
                    print(BPE.decode(batch['a'][-2][:batch['a_len'][-2]]))
                    print(BPE.decode(batch['c'][-1][:batch['c_len'][-1]]))
                    print(BPE.decode(batch['a'][-1][:batch['a_len'][-1]]))
                    print(q_golds[-2:])
                    print(q_preds[-2:])
                    # print(BPE.decode(greedy_output.data[0][:greedy_output_lens[0]]))

            test_loss /= num_batches
            self.logger.info('Dev set: Average loss: {:.4f}'.format(test_loss))

        dev_bleu = 100*bleu_corpus(q_golds, q_preds)

        add_to_log('dev/loss', test_loss, self.current_iteration, self.config.tag +'/' + self.run_id)
        add_to_log('dev/bleu', dev_bleu, self.current_iteration, self.config.tag +'/' + self.run_id)
        
        self.logger.info('BLEU: {:}'.format(dev_bleu))

        # TODO: refactor this out somewhere
        if self.best_metric is None \
                or test_loss < self.best_metric \
                or force_save_output \
                or (self.config.training.early_stopping_lag > 0 and self.best_epoch is not None and (self.current_epoch-self.best_epoch) <= self.config.training.early_stopping_lag > 0):
            with open(os.path.join(FLAGS.output_path, self.config.tag, self.run_id,'output.txt'), 'w') as f:
                f.write("\n".join(q_preds))

            
            self.all_metrics_at_best = {
                'bleu': dev_bleu,
                'nll': test_loss.item()
            }

            with open(os.path.join(FLAGS.output_path, self.config.tag, self.run_id,'metrics.json'), 'w') as f:
                json.dump(self.all_metrics_at_best, f)

        if self.best_metric is None \
                or test_loss < self.best_metric \
                or (self.config.training.early_stopping_lag > 0 and self.best_epoch is not None and (self.current_epoch-self.best_epoch) <= self.config.training.early_stopping_lag > 0):
            

            if test_loss < self.best_metric:
                self.logger.info('New best score! Saving...')
            else:
                self.logger.info('Early stopping lag active: saving...')
                
            self.best_metric = test_loss
            self.best_epoch = self.current_epoch

            self.save_checkpoint()

        


