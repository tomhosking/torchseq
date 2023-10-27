import logging

import torch

# import torch._dynamo

from torchseq.utils.functions import to_device_unless_marked

# This project was originally based off this template:
# https://github.com/moemen95/Pytorch-Project-Template


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    cuda: bool
    use_lightning: bool = False

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def set_device(self, use_cuda=True):
        # set cuda flag
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available and not use_cuda:
            self.logger.warning("You have a CUDA device, so you should probably enable CUDA")

        if use_cuda and not self.cuda_available:
            self.logger.error("Use CUDA is set to true, but not CUDA devices were found!")
            raise Exception("No CUDA devices found")

        self.cuda = self.cuda_available & use_cuda

        if not self.model:
            raise Exception("You need to define your model before calling set_device!")

        if self.cuda:
            self.device = torch.device("cuda")

            self.logger.info("Model will run on *****GPU-CUDA***** ")

            # self.model.to(self.device)
            self.model.apply(to_device_unless_marked(self.device))

            self.loss.to(self.device)

            # TODO: Enable for pytorch 2.0
            # torch._dynamo.config.verbose = False
            # torch._dynamo.config.log_level = logging.WARN
            # torch._dynamo.reset()
            # self.model = torch.compile(self.model, dynamic=True, mode='reduce-overhead')  #, backend="inductor",fullgraph=True, mode='reduce-overhead',, dynamic=True

        else:
            self.device = torch.device("cpu")

            self.logger.info("Model will run on *****CPU*****")

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pt", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def train(self, data_loader) -> None:
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
