import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchseq.datasets.qa_dataset import QADataset
from torchseq.utils.seed import init_worker
from torchseq.utils.tokenizer import Tokenizer


class PreprocessedDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        train = torch.load(os.path.join(config.env.data_path, "processed", "train_processed.pt"))
        valid = torch.load(os.path.join(config.env.data_path, "processed", "valid_processed.pt"))

        self.len_train_data = len(train)
        self.len_valid_data = len(valid)

        self.train_iterations = (
            self.len_train_data + self.config.training.batch_size - 1
        ) // self.config.training.batch_size
        self.valid_iterations = (
            self.len_valid_data + self.config.training.batch_size - 1
        ) // self.config.training.batch_size

        self.train_loader = DataLoader(
            train,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self.pad_and_order_sequences,
            worker_init_fn=init_worker,
        )

        self.valid_loader = DataLoader(
            valid,
            batch_size=config.eval.eval_batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=self.pad_and_order_sequences,
            worker_init_fn=init_worker,
        )

    def pad_and_order_sequences(self, batch):
        keys = batch[0].keys()
        max_lens = {k: max(len(x[k]) for x in batch) for k in keys}

        for x in batch:
            for k in keys:
                if k == "a_pos":
                    x[k] = F.pad(x[k], (0, max_lens[k] - len(x[k])), value=0)
                elif k[-5:] != "_text":
                    x[k] = F.pad(x[k], (0, max_lens[k] - len(x[k])), value=Tokenizer().pad_id)

        tensor_batch = {}
        for k in keys:
            tensor_batch[k] = torch.stack([x[k] for x in batch], 0).squeeze()

        return tensor_batch
