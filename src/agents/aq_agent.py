import numpy as np

from tqdm import tqdm
import shutil
import random
import json


import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from agents.base import BaseAgent

from models.aq_transformer import TransformerAqModel

from datasets.squad_loader import SquadDataLoader
from datasets.preprocessed_loader import PreprocessedDataLoader
from datasets.loaders import load_glove, get_embeddings

from utils.logging import add_to_log

from utils.misc import print_cuda_statistics
from utils.metrics import bleu_corpus

from utils.bpe_factory import BPE

from models.samplers.greedy import GreedySampler
from models.samplers.beam_search import BeamSearchSampler
from models.samplers.teacher_force import TeacherForcedSampler

import os

cudnn.benchmark = True


class AQAgent(BaseAgent):

    def __init__(self, config, run_id):
        super().__init__(config)

        self.run_id = run_id


        os.makedirs('./runs/' + run_id + '/model/')
        with open('./runs/' + run_id +'/config.json', 'w') as f:
            json.dump(config.data, f)

        # load glove embeddings
        # all_glove = load_glove(config.env.data_path+'/', d=config.embedding_dim)
        # glove_init = get_embeddings(glove=all_glove, vocab=vocab, D=config.embedding_dim)

        # define models
        self.model = TransformerAqModel(config)

        # define data_loader
        if self.config.training.use_preprocessed_data:
            self.data_loader = PreprocessedDataLoader(config=config)
        else:
            self.data_loader = SquadDataLoader(config=config)

        # define loss
        self.loss = nn.CrossEntropyLoss(ignore_index=BPE.pad_id, reduction='none')
        # self.loss = CrossEntropyLossWithLS(ignore_index=config.prepro.vocab_size, smooth_eps=config.label_smoothing, )

        # define optimizer
        if config.training.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.lr, betas=(0.9, 0.98), eps=1e-9)
        elif config.training.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.training.lr)
        else:
            raise Exception("Unrecognised optimiser: " + config.training.opt)

        # initialize counter
        self.best_metric = None
        self.current_epoch = 0
        self.current_iteration = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.env.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.env.cuda

        # set the manual seed for torch
        # self.manual_seed = self.config.seed
        if self.cuda:
            # torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.env.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            # print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            # torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        self.model.device = self.device

        self.decode_greedy = GreedySampler(config, self.device)
        self.decode_beam = BeamSearchSampler(config, self.device)
        self.decode_teacher_force = TeacherForcedSampler(config, self.device)


    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        self.logger.info('Loading from checkpoint ' + file_name)
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.best_metric = checkpoint['best_metric']
        

    def save_checkpoint(self, file_name="checkpoint.pth.tar"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """

        torch.save(
            {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'best_metric': self.best_metric
            },
            './runs/' + self.run_id + '/model/' + file_name)

    def train(self):
        """
        Main training loop
        :return:
        """
        self.global_idx = self.current_epoch * len(self.data_loader.train_loader.dataset)
        for epoch in range(self.config.training.num_epochs):
            self.train_one_epoch()

            self.current_epoch += 1
            if (epoch+1) % 5 == 0 or True:
                self.validate(save=True)

    
    def get_lr(self, step):
        if self.config.training.lr_schedule:
            step = max(step, 1)
            return pow(300, -0.5) * min(pow(step, -0.5), step * pow(5000, -1.5))
        else:
            return self.config.training.lr
        

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        self.model.train()
        
        for batch_idx, batch in enumerate(tqdm(self.data_loader.train_loader, desc='Epoch {:}'.format(self.current_epoch))):
            batch= {k:v.to(self.device) for k,v in batch.items()}
            curr_batch_size = batch['c'].size()[0]
            max_output_len = batch['q'].size()[1]
            self.global_idx += curr_batch_size

            self.optimizer.zero_grad()
            loss = 0

            
            output, logits = self.decode_teacher_force(self.model, batch)
            
            
            # print(logits.shape)
            # q_gold_mask = (torch.arange(max_output_len)[None, :].cpu() > batch['q_len'][:, None].cpu()).to(self.device)
            this_loss = self.loss(logits.permute(0,2,1), batch['q'])
            
            loss += torch.mean(torch.sum(this_loss, dim=1)/batch['q_len'].to(this_loss), dim=0)
            # print(loss)
            lr = self.get_lr(self.current_iteration)
            add_to_log('train/lr', lr, self.current_iteration, self.run_id)
            loss.lr = lr
            loss.backward()
            if batch_idx  == 0:
                for n,p in self.model.named_parameters():
                    if p.requires_grad and 'embeddings' in n:
                        print('{}: {}'.format(n,torch.sum(p.grad)))
            # exit()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_gradient)
            self.optimizer.step()
            if batch_idx % self.config.training.log_interval == 0:
                add_to_log('train/loss', loss, self.current_iteration, self.run_id)
                
                # # print(batch['q'][0])
                # # print(greedy_output.data[0])
                if batch_idx % (self.config.training.log_interval*20) == 0:

                    with torch.no_grad():
                        greedy_output, _, output_lens = self.decode_greedy(self.model, batch)
                    print(BPE.instance().decode_ids(batch['q'][0][:batch['q_len'][0]]))
                    print(BPE.instance().decode_ids(greedy_output.data[0][:output_lens[0]]))
                # self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     self.current_epoch, batch_idx * curr_batch_size, len(self.data_loader.train_loader.dataset),
                #            100. * batch_idx / len(self.data_loader.train_loader), loss.item()))

                torch.cuda.empty_cache()
            self.current_iteration += 1

    def validate(self, save=False, force_save_output=False):
        """
        One cycle of model validation
        :return:
        """
        print('## Validating')
        self.model.eval()
        test_loss = 0
        correct = 0

        q_preds = []
        q_golds = []

        with torch.no_grad():
            num_batches = 0
            for batch_idx, batch in enumerate(tqdm(self.data_loader.valid_loader, desc='Validating after {:} epochs'.format(self.current_epoch))):
                batch= {k:v.to(self.device) for k,v in batch.items()}
                curr_batch_size = batch['c'].size()[0]
                max_output_len = batch['q'].size()[1]

                # loss = 0

                _, logits = self.decode_teacher_force(self.model, batch)
                
                # dev_output, _, dev_output_lens = self.decode_greedy(self.model, batch)
                beam_output, _, beam_lens = self.decode_beam(self.model, batch)

                # if batch_idx == 0:
                #     print('OUTPUT OF BEAM')
                #     for i in range(beam_output.shape[1]):
                #         print(BPE.instance().decode_ids(beam_output.data[0,i][:output_lens[0,i]]))
                #     exit()

                dev_output = beam_output[:, 0, :]
                dev_output_lens = beam_lens[:, 0]

                this_loss = self.loss(logits.permute(0,2,1), batch['q'])
            
                test_loss += torch.mean(torch.sum(this_loss, dim=1)/batch['q_len'].to(this_loss), dim=0)
                num_batches += 1

                for ix, q_pred in enumerate(dev_output.data):
                    q_preds.append(BPE.instance().decode_ids(q_pred[:dev_output_lens[ix]]))
                for ix, q_gold in enumerate(batch['q']):
                    q_golds.append(BPE.instance().decode_ids(q_gold[:batch['q_len'][ix]]))
                

                if batch_idx % 200 == 0:
                    # print(batch['c'][0][:10])
                    # for ix in range(batch['c_len'][-2]):
                    #     print(" ".join([BPE.instance().pieces[batch['c'][-2][ix]] +'//' + str(batch['a_pos'][-2][ix])]))
                    print(BPE.instance().decode_ids(batch['c'][-2][:batch['c_len'][-2]]))
                    print(BPE.instance().decode_ids(batch['a'][-2][:batch['a_len'][-2]]))
                    print(BPE.instance().decode_ids(batch['c'][-1][:batch['c_len'][-1]]))
                    print(BPE.instance().decode_ids(batch['a'][-1][:batch['a_len'][-1]]))
                    print(q_golds[-2:])
                    print(q_preds[-2:])
                    # print(BPE.instance().decode_ids(greedy_output.data[0][:greedy_output_lens[0]]))

            test_loss /= num_batches
            self.logger.info('Dev set: Average loss: {:.4f}'.format(test_loss))

        dev_bleu = 100*bleu_corpus(q_golds, q_preds)

        add_to_log('dev/loss', test_loss, self.current_iteration, self.run_id)
        add_to_log('dev/bleu', dev_bleu, self.current_iteration, self.run_id)
        
        self.logger.info('BLEU: {:}'.format(dev_bleu))

        if self.best_metric is None or test_loss < self.best_metric:
            self.best_metric = test_loss

            self.logger.info('New best score! Saving...')
            self.save_checkpoint()

        # TODO: refactor this out somewhere
        if self.best_metric is None or test_loss < self.best_metric or force_save_output:
            with open('./runs/' + self.run_id +'/output.txt', 'w') as f:
                f.writelines(q_preds)
