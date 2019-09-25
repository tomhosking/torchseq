"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from agents.base import BaseAgent

from models.aq_transformer import TransformerAqModel
from datasets.squad_loader import SquadDataLoader
from datasets.loaders import load_glove, get_embeddings

from utils.logging import add_to_log

from utils.misc import print_cuda_statistics

from utils.bpe_factory import BPE

cudnn.benchmark = True


class AQAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)


        # load glove embeddings
        # all_glove = load_glove(config.data_path+'/', d=config.embedding_dim)
        # glove_init = get_embeddings(glove=all_glove, vocab=vocab, D=config.embedding_dim)

        # define models
        self.model = TransformerAqModel(config)

        # define data_loader
        self.data_loader = SquadDataLoader(config=config)

        # define loss
        self.loss = nn.CrossEntropyLoss(ignore_index=10000)

        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        self.model.device = self.device

        # Model Loading from the latest checkpoint if not found start from scratch.
        # self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        # self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass


    # TODO: These should be used to actually generate output from the model, and return a loss (and the output)
    def teacher_force(self):
        pass


    def sample(self):
        pass


    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, 1 + 1):
            self.train_one_epoch()
            self.validate()

            self.current_epoch += 1
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        self.model.train()
        global_idx = 0
        for batch_idx, batch in enumerate(self.data_loader.train_loader):
            batch= {k:v.to(self.device) for k,v in batch.items()}
            curr_batch_size = batch['c'].size()[0]
            max_output_len = batch['q'].size()[1]
            global_idx += curr_batch_size

            self.optimizer.zero_grad()
            loss = 0

            output = torch.LongTensor(curr_batch_size, batch['q'].size()[1]).zero_().to(self.device)
            
            for seq_ix in range(max_output_len):
                logits = self.model(batch, output)  
                
                output = torch.argmax(logits, -1)
            
            
            # print(logits.shape)
            q_gold_mask = (torch.arange(max_output_len)[None, :].cpu() > batch['q_len'][:, None].cpu()).to(self.device)

            loss += self.loss(logits.permute(0,2,1), batch['q'])
            # print(loss)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                add_to_log('train/loss', loss, global_idx)
                
                print(batch['q'][0])
                print(output.data[0])
                print(BPE.instance().decode_ids(batch['q'][0][:batch['q_len'][0]]))
                print(BPE.instance().decode_ids(output.data[0][:batch['q_len'][0]]))
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * curr_batch_size, len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.data_loader.test_loader.dataset)
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data_loader.test_loader.dataset),
            100. * correct / len(self.data_loader.test_loader.dataset)))
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
