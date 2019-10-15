"""
Mnist Main agent, as mentioned in the tutorial
"""
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
from models.cross_entropy import CrossEntropyLossWithLS
from datasets.squad_loader import SquadDataLoader
from datasets.loaders import load_glove, get_embeddings

from utils.logging import add_to_log

from utils.misc import print_cuda_statistics
from utils.metrics import bleu_corpus

from utils.bpe_factory import BPE

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
        # all_glove = load_glove(config.data_path+'/', d=config.embedding_dim)
        # glove_init = get_embeddings(glove=all_glove, vocab=vocab, D=config.embedding_dim)

        # define models
        self.model = TransformerAqModel(config)

        # define data_loader
        self.data_loader = SquadDataLoader(config=config)

        # define loss
        self.loss = nn.CrossEntropyLoss(ignore_index=BPE.pad_id, reduction='none')
        # self.loss = CrossEntropyLossWithLS(ignore_index=config.vocab_size, smooth_eps=config.label_smoothing, )

        # define optimizer
        if config.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, betas=(0.9, 0.98), eps=1e-9)
        elif config.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)
        else:
            raise Exception("Unrecognised optimiser: " + config.opt)

        # initialize counter
        self.best_metric = None
        self.current_epoch = 0
        self.current_iteration = 0

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
            # print_cuda_statistics()
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


    # TODO: These should be used to actually generate output from the model, and return a loss (and the output)
    
    def decode_teacher_force(self, model, batch):
        curr_batch_size = batch['c'].size()[0]
        max_output_len = batch['q'].size()[1]

        # Create vector of SOS + placeholder for first prediction
        
        
        logits = torch.FloatTensor(curr_batch_size, 1, self.config.vocab_size+1).fill_(float('-1e6')).to(self.device)
        logits[:, :, BPE.instance().BOS] = float('1e6')

        # With a transformer decoder, we can lean on the internal mask to ensure that the model can't see ahead
        # ..and then just do a single pass through the whole model using the gold output as input
        output = batch['q'][:, :max_output_len-1].to(self.device)
        pred_logits = model(batch, output)

        logits = torch.cat([logits, pred_logits], dim=1)

        return output, logits

    
    def decode_greedy(self, model, batch):
        curr_batch_size = batch['c'].size()[0]
        max_output_len = batch['q'].size()[1]

        # Create vector of SOS + placeholder for first prediction
        output = torch.LongTensor(curr_batch_size, 1).fill_(BPE.instance().BOS).to(self.device)
        logits = torch.FloatTensor(curr_batch_size, 1, self.config.vocab_size+1).fill_(float('-inf')).to(self.device)
        logits[:, :, BPE.instance().BOS] = float('inf')

        output_done = torch.BoolTensor(curr_batch_size).fill_(False).to(self.device)
        padding = torch.LongTensor(curr_batch_size).fill_(self.config.vocab_size).to(self.device)

        seq_ix = 0
        while torch.sum(output_done) < curr_batch_size and seq_ix < max_output_len:
            new_logits = model(batch, output)

            new_output = torch.argmax(new_logits, -1)

            new_scores = torch.max(new_logits, -1).values
            # print(new_scores)

            # Use pad for the output for elements that have completed
            new_output[:, -1] = torch.where(output_done, padding, new_output[:, -1])
            
            output = torch.cat([output, new_output[:, -1].unsqueeze(-1)], dim=-1)

            logits = torch.cat([logits, new_logits[:, -1:, :]], dim=1)

            # print(output_done)
            # print(output[:, -1])
            # print(output[:, -1] == BPE.instance().EOS)
            output_done = output_done | (output[:, -1] == BPE.instance().EOS)
            seq_ix += 1
        # exit()
        
        # output.where(output == BPE.pad_id, torch.LongTensor(output.shape).fill_(-1).to(self.device))

        return output, logits, torch.sum(output != BPE.pad_id, dim=-1)

    def decode_beam(self, model, batch):
        curr_batch_size = batch['c'].size()[0]
        max_output_len = batch['q'].size()[1]

        # TODO: move to config
        beam_width = 8 # number of total hypotheses to maintain
        beam_expansion = 3 # number of new predictions to add to each hypothesis each step


        # Create vector of SOS + placeholder for first prediction
        output_seq = torch.LongTensor(curr_batch_size, beam_width, 1).fill_(BPE.instance().BOS).to(self.device)
        scores = torch.FloatTensor(curr_batch_size, beam_width, 1).fill_(0).to(self.device)
        

        output_done = torch.BoolTensor(curr_batch_size, beam_width).fill_(False).to(self.device)
        padding = torch.LongTensor(curr_batch_size, beam_width).fill_(self.config.vocab_size).to(self.device)
        pad_probs = torch.FloatTensor(curr_batch_size, beam_width, self.config.vocab_size+1).fill_(float('-inf')).to(self.device)
        pad_probs[:,:,BPE.pad_id] = float('0')

        
        def _tile_batch(x):
            return x.repeat_interleave(beam_width, dim=0)

        batch_tiled = {k: _tile_batch(x) for k,x in batch.items()}

        seq_ix = 0
        while torch.sum(output_done) < curr_batch_size*beam_width and seq_ix < max_output_len:
            
            new_logits = model(batch_tiled, output_seq.view(curr_batch_size*beam_width, -1)).view(curr_batch_size, beam_width, -1, self.config.vocab_size+1)
            output_done = (output_seq[:,:,-1] == BPE.pad_id) | (output_seq[:,:,-1] == BPE.instance().EOS)
            # print(output_done.shape)
            # print(output_done.unsqueeze(-1).shape)
            # print(pad_probs.shape)
            # print(new_logits.shape)
            new_probs = torch.where(output_done.unsqueeze(-1), pad_probs, torch.log_softmax(new_logits[:, :, -1, :], -1))

            if seq_ix == 0:
                top_expansions = torch.topk(new_probs, k=beam_width, dim=-1, largest=True)

                # print(output_seq.shape)
                # print(top_expansions.indices.shape)
                
                # On first iteration, the beams are all the same! So spread the topk across beams
                output_seq = torch.cat([output_seq, top_expansions.indices.unsqueeze(2)[:, 0, :, :].permute(0,2,1)], dim=-1)
                scores = torch.cat([scores, top_expansions.values.unsqueeze(2)[:, 0, :, :].permute(0,2,1)], dim=-1)
                # print(scores)
                # exit()
            else:

                top_expansions = torch.topk(new_probs, k=beam_expansion, dim=-1, largest=True)
                
                # print(new_probs.shape)
                # print(output_seq.shape)
                # print(scores.shape)
                # print(top_expansions.indices.shape)
                # print(top_expansions.values.shape)
                
                

                expanded_beam_ixs = torch.cat([output_seq.unsqueeze(-2).expand(-1, -1, beam_expansion, -1), top_expansions.indices.unsqueeze(-1)], dim=-1)
                expanded_beam_scores = torch.cat([scores.unsqueeze(-2).expand(-1, -1, beam_expansion, -1), top_expansions.values.unsqueeze(-1)], dim=-1)

                

                curr_seq_len = expanded_beam_ixs.shape[3]

                expanded_beam_scores = expanded_beam_scores.view(curr_batch_size, beam_width*beam_expansion, curr_seq_len)
                expanded_beam_ixs = expanded_beam_ixs.view(curr_batch_size, beam_width*beam_expansion, curr_seq_len)


                # print(expanded_beam_ixs.shape)
                # print(expanded_beam_scores)

                beam_scores = torch.sum(expanded_beam_scores, dim=-1)

                # print(beam_scores.shape)

                top_beams = torch.topk(beam_scores, k=beam_width, dim=-1)


                # print(top_beams.indices.shape)
                # print(expanded_beam_scores.shape)


                scores = torch.gather(expanded_beam_scores, 1, top_beams.indices.unsqueeze(-1).expand(-1,-1, curr_seq_len))
                new_output = torch.gather(expanded_beam_ixs, 1, top_beams.indices.unsqueeze(-1).expand(-1,-1, curr_seq_len))


                # print(new_output.shape)
                # how to get token_ix and curr prob from the top beam?

                # Use pad for the output for elements that have completed
                output_done = (new_output[:, :, -2] == BPE.instance().EOS) | (new_output[:, :, -2] == BPE.pad_id)
                new_output[:, :, -1] = torch.where(output_done, padding, new_output[:, :, -1])
                
                # output_seq = torch.cat([output_seq, new_output], dim=-1)
                output_seq = new_output
                # scores = torch.cat([scores, new_scores], dim=-1)

                # print(output_done)
                # exit()
                
                # exit()

            
            seq_ix += 1
        
        # Take top-1 beam
        # output = output_seq.view(curr_batch_size, beam_width, -1)[:, 0, :]

        # output_seq = torch.where(output_seq == BPE.pad_id, torch.LongTensor(output_seq.shape).fill_(-1).to(self.device), output_seq)
        # print(output_seq)
        return output_seq, None, torch.sum(output_seq != BPE.pad_id, dim=-1)



    def train(self):
        """
        Main training loop
        :return:
        """
        self.global_idx = self.current_epoch * len(self.data_loader.train_loader.dataset)
        for epoch in range(self.config.num_epochs):
            self.train_one_epoch()

            self.current_epoch += 1
            if (epoch+1) % 5 == 0 or True:
                self.validate(save=True)

    @staticmethod
    def get_lr(step):
        return 1e-5
        step = max(step, 1)
        return pow(300, -0.5) * min(pow(step, -0.5), step * pow(5000, -1.5))

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
            lr = AQAgent.get_lr(self.current_iteration)
            add_to_log('train/lr', lr, self.current_iteration, self.run_id)
            loss.lr = lr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_gradient)
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                add_to_log('train/loss', loss, self.current_iteration, self.run_id)
                
                # # print(batch['q'][0])
                # # print(greedy_output.data[0])
                if batch_idx % (self.config.log_interval*20) == 0:

                    with torch.no_grad():
                        greedy_output, _, output_lens = self.decode_greedy(self.model, batch)
                    print(BPE.instance().decode_ids(batch['q'][0][:batch['q_len'][0]]))
                    print(BPE.instance().decode_ids(greedy_output.data[0][:output_lens[0]]))
                # self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     self.current_epoch, batch_idx * curr_batch_size, len(self.data_loader.train_loader.dataset),
                #            100. * batch_idx / len(self.data_loader.train_loader), loss.item()))

                torch.cuda.empty_cache()
            self.current_iteration += 1

    def validate(self, save=False):
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
                    print(BPE.instance().decode_ids(batch['c'][0][:batch['c_len'][0]]))
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

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
