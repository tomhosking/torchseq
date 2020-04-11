import logging

import torch

from utils.parallel import DataParallelCriterion, DataParallelModel


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def set_device(self):
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
            # torch.cuda.set_device(self.config.env.gpu_device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            # print_cuda_statistics()

            if torch.cuda.device_count() > 1:
                self.logger.info("Multi GPU available: using {:} GPUs!".format(torch.cuda.device_count()))

            # self.model = DataParallelModel(self.model)
            # self.loss = DataParallelCriterion(self.loss)

            # self.model.module = self.model

            self.model.to(self.device)
            self.loss.to(self.device)

        else:
            self.device = torch.device("cpu")
            # torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        if not self.model:
            raise Exception("You need to define your model before calling set_device!")

        self.model.device = self.device
        # if torch.cuda.device_count() > 1 or True:
        # #     # DataParallel hides the 'real' model, so we have to set it manually
        #     self.model.module.device = self.device

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError
