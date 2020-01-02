from agents.base import BaseAgent

from args import FLAGS as FLAGS

import os, json
import torch

import torch.optim as optim

from models.samplers.greedy import GreedySampler
from models.samplers.beam_search import BeamSearchSampler
from models.samplers.teacher_force import TeacherForcedSampler
from models.samplers.parallel_nucleus import ParallelNucleusSampler

class ModelAgent(BaseAgent):


    def __init__(self, config, run_id, silent=False):
        super().__init__(config)

        self.run_id = run_id
        self.silent = silent


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