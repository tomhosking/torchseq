# from torch.utils.tensorboard import SummaryWriter

from torchseq.utils.singleton import Singleton

import os
import torch


from torchseq.utils.wandb import wandb_log
from wandb import Histogram as wbHistogram


class Logger(metaclass=Singleton):
    writer = None
    step = 0

    def __init__(self, silent=False, log_path=None, interval=10):
        self.silent = silent
        self.interval = interval

        # if log_path is not None:
        #     self.writer = SummaryWriter(log_path)

    def log_scalar(self, key, value, iteration):
        # if iteration < self.step:
        #     raise Exception("What's the first thing that decreases step?!")
        self.step = iteration
        if iteration % self.interval != 0:
            return
        wandb_log({key: value}, step=iteration)
        if self.writer is not None:
            self.writer.add_scalar(key, value, iteration)

    def log_histogram(self, key, value, iteration):
        if max(value) >= 512:
            value = [x for x in value if x < 512]
        if len(value) == 0:
            return

        wandb_log({key: wbHistogram(value, num_bins=int(max(value)) + 1)}, step=iteration)
        if self.writer is not None:
            self.writer.add_histogram(key, torch.Tensor(value), iteration)

    def log_text(self, key, value, iteration):
        if self.writer is not None:
            self.writer.add_text(key, value, iteration)
