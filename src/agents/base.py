import logging

import torch


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

        if self.cuda:
            self.device = torch.device("cuda")

            self.logger.info("Program will run on *****GPU-CUDA***** ")

            if torch.cuda.device_count() > 1:
                self.logger.info("Multi GPU available: using {:} GPUs!".format(torch.cuda.device_count()))

            self.model.to(self.device)
            self.loss.to(self.device)

        else:
            self.device = torch.device("cpu")

            self.logger.info("Program will run on *****CPU*****\n")

        if not self.model:
            raise Exception("You need to define your model before calling set_device!")

        self.model.device = self.device

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
