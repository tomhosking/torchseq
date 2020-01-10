from agents.base import BaseAgent

from args import FLAGS as FLAGS

from tqdm import tqdm

import os, json
import torch

import torch.optim as optim

from models.samplers.greedy import GreedySampler
from models.samplers.beam_search import BeamSearchSampler
from models.samplers.teacher_force import TeacherForcedSampler
from models.samplers.parallel_nucleus import ParallelNucleusSampler

from utils.mckenzie import update_mckenzie

from models.lr_schedule import get_lr
from utils.logging import add_to_log

from utils.bpe_factory import BPE

from utils.metrics import bleu_corpus


# Variable length sequences = worse performance if we try to optimise
from torch.backends import cudnn
cudnn.benchmark = False

class ModelAgent(BaseAgent):


    def __init__(self, config, run_id, silent=False):
        super().__init__(config)

        self.run_id = run_id
        self.silent = silent

        # Slightly hacky way of allowing for inference-only use
        if run_id is not None:
            os.makedirs(os.path.join(FLAGS.output_path, self.config.tag, self.run_id, 'model'))
            with open(os.path.join(FLAGS.output_path, self.config.tag, self.run_id, 'config.json'), 'w') as f:
                json.dump(config.data, f)

        

        # initialize counter
        self.best_metric = None
        self.all_metrics_at_best = {}
        self.best_epoch = None
        self.current_epoch = 0
        self.current_iteration = 0

    def create_optimizer(self):
        if self.config.training.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.lr, betas=(self.config.training.beta1, self.config.training.beta2), eps=1e-9)
        elif config.training.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.training.lr)
        else:
            raise Exception("Unrecognised optimiser: " + self.config.training.opt)

    def create_samplers(self):
        self.decode_greedy = GreedySampler(self.config, self.device)
        self.decode_beam = BeamSearchSampler(self.config, self.device)
        self.decode_teacher_force = TeacherForcedSampler(self.config, self.device)
        self.decode_nucleus = ParallelNucleusSampler(self.config, self.device)

    
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
            os.path.join(FLAGS.output_path, self.config.tag, self.run_id, 'model', file_name))

    
    def begin_epoch_hook(self):
        """
        Override this if you want any logic to be triggered at the start of a new training epoch
        """
        pass

    def text_to_batch(self, input):
        """
        Convert a dictionary of strings to a batch that can be used as input to the model
        """
        raise NotImplementedError("Your model is missing a text_to_batch method!")

    def infer(self, input):
        """
        Run inference on a dictionary of strings
        """
        batch = self.text_to_batch(input, self.device)

        _, output, output_lens = self.step_validate(batch, self.tgt_field, sample_outputs=True, calculate_loss=False)

        output_strings = [BPE.decode(output.data[ix][:output_lens[ix]]) for ix in range(len(output_lens))]

        return output_strings

    # def step_train(self, batch):
    #     """
    #     Implement the main logic here for a single training step
    #     """
    #     pass

    # def step_validate(self, batch):
    #     """
    #     Implement the main logic here for a single validation/inference step
    #     """
    #     pass


        
    def step_train(self, batch, tgt_field):
        loss = 0
            
        output, logits = self.decode_teacher_force(self.model, batch, tgt_field)

        this_loss = self.loss(logits.permute(0,2,1), batch[tgt_field])

        loss += torch.mean(torch.sum(this_loss, dim=1)/batch[tgt_field+'_len'].to(this_loss), dim=0)

        return loss

    def train(self):
        """
        Main training loop
        :return:
        """
        if self.tgt_field is None:
            raise Exception('You need to specify the target output field! ie which element of a batch is the output')

        self.global_idx = self.current_epoch * len(self.data_loader.train_loader.dataset)
        self.current_iteration = self.current_epoch * len(self.data_loader.train_loader.dataset)//self.config.training.optim_batch_size
        update_mckenzie(0,'-')

        # If we're starting above zero, means we've loaded from chkpt -> validate to give a starting point for fine tuning
        if self.current_epoch > 0:
            self.begin_epoch_hook()
            self.validate(save=True, tgt_field=self.tgt_field)

        for epoch in range(self.config.training.num_epochs):
            self.begin_epoch_hook()
            
            self.train_one_epoch()

            self.current_epoch += 1

            if self.current_epoch > self.config.training.warmup_epochs:
                self.validate(save=True, tgt_field=self.tgt_field)



            if 'bleu' in self.all_metrics_at_best:
                update_mckenzie((epoch+1)/self.config.training.num_epochs*100, "{:0.2f}".format(self.all_metrics_at_best['bleu']))
            else:
                update_mckenzie((epoch+1)/self.config.training.num_epochs*100, "-")

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
            curr_batch_size = batch[[k for k in batch.keys()][0]].size()[0]
            
            self.global_idx += curr_batch_size
            
            # Weight the loss by the ratio of this batch to optimiser step size, so that LR is equivalent even when grad accumulation happens
            loss = self.step_train(batch, self.tgt_field) * float(curr_batch_size)/float(self.config.training.optim_batch_size)
            
            
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
                
                # TODO: This is currently paraphrase specific! May work for other models but isn't guaranteed
                if batch_idx % (self.config.training.log_interval*20) == 0 and not self.silent:

                    with torch.no_grad():
                        greedy_output, _, output_lens = self.decode_greedy(self.model, batch, self.tgt_field)
                    
                    # print(BPE.decode(batch['s1'][0][:batch['s1_len'][0]]))
                    print(BPE.decode(batch[self.tgt_field][0][:batch[self.tgt_field+'_len'][0]]))
                    print(BPE.decode(greedy_output.data[0][:output_lens[0]]))

                torch.cuda.empty_cache()


    def step_validate(self, batch, tgt_field, sample_outputs=True, calculate_loss=True):
        
        if not sample_outputs:
            dev_output = None
            dev_output_lens = None
        elif self.config.eval.sampler == 'nucleus':
            dev_output, _, dev_output_lens = self.decode_nucleus(self.model, batch, tgt_field)
        elif self.config.eval.sampler == 'beam':
            beam_output, _, beam_lens = self.decode_beam(self.model, batch, tgt_field)

            dev_output = beam_output[:, 0, :]
            dev_output_lens = beam_lens[:, 0]
        else:
            raise Exception("Unknown sampling method!")

        if calculate_loss:
            _, logits = self.decode_teacher_force(self.model, batch, tgt_field)
            this_loss = self.loss(logits.permute(0,2,1), batch[tgt_field])
            normed_loss = torch.mean(torch.sum(this_loss, dim=1)/batch[tgt_field+'_len'].to(this_loss), dim=0)
        else:
            normed_loss = None
        return normed_loss, dev_output, dev_output_lens

    def validate(self, save=False, force_save_output=False, use_test=False, tgt_field=None):
        """
        One cycle of model validation
        :return:
        """

        if tgt_field is None:
            raise Exception('You need to specify the target output field! ie which element of a batch is the output')

        self.logger.info('## Validating after {:} epochs'.format(self.current_epoch))
        self.model.eval()
        test_loss = 0
        correct = 0

        pred_output = []
        gold_output = []

        if use_test:
            print('***** USING TEST SET ******')
            valid_loader = self.data_loader.test_loader
        else:
            valid_loader = self.data_loader.valid_loader

        with torch.no_grad():
            num_batches = 0
            for batch_idx, batch in enumerate(tqdm(valid_loader, desc='Validating after {:} epochs'.format(self.current_epoch), disable=self.silent)):
                batch= {k:v.to(self.device) for k,v in batch.items()}
                curr_batch_size = batch[[k for k in batch.keys()][0]].size()[0]
                
                this_loss, dev_output, dev_output_lens = self.step_validate(batch, self.tgt_field)

                test_loss += this_loss

                num_batches += 1

                for ix, pred in enumerate(dev_output.data):
                    pred_output.append(BPE.decode(pred[:dev_output_lens[ix]]))
                for ix, gold in enumerate(batch[tgt_field]):
                    gold_output.append(BPE.decode(gold[:batch[tgt_field+'_len'][ix]]))
                

                if batch_idx % 200 == 0 and not self.silent:
                    print(gold_output[-2:])
                    print(pred_output[-2:])

            test_loss /= num_batches
            self.logger.info('Dev set: Average loss: {:.4f}'.format(test_loss))

        dev_bleu = 100*bleu_corpus(gold_output, pred_output)

        add_to_log('dev/loss', test_loss, self.current_iteration, self.config.tag +'/' + self.run_id)
        add_to_log('dev/bleu', dev_bleu, self.current_iteration, self.config.tag +'/' + self.run_id)
        
        self.logger.info('BLEU: {:}'.format(dev_bleu))

        

        # TODO: sort this out - there's got to be a more compact way of doing it all
        if self.best_metric is None \
                or test_loss < self.best_metric \
                or force_save_output \
                or (self.config.training.early_stopping_lag > 0 and self.best_epoch is not None and (self.current_epoch-self.best_epoch) <= self.config.training.early_stopping_lag > 0):
            with open(os.path.join(FLAGS.output_path, self.config.tag, self.run_id,'output.txt'), 'w') as f:
                f.write("\n".join(pred_output))

            
            self.all_metrics_at_best = {
                'bleu': dev_bleu,
                'nll': test_loss.item()
            }

            with open(os.path.join(FLAGS.output_path, self.config.tag, self.run_id,'metrics.json'), 'w') as f:
                json.dump(self.all_metrics_at_best, f)

        if self.best_metric is None \
                or test_loss < self.best_metric \
                or (self.config.training.early_stopping_lag > 0 and self.best_epoch is not None and (self.current_epoch-self.best_epoch) <= self.config.training.early_stopping_lag > 0):
            

            if self.best_metric is None:
                self.best_epoch = self.current_epoch
                self.best_metric = test_loss
            elif test_loss < self.best_metric:
                self.logger.info('New best score! Saving...')
                self.best_epoch = self.current_epoch
                self.best_metric = test_loss
            else:
                self.logger.info('Early stopping lag active: saving...')
                
            

            self.save_checkpoint()