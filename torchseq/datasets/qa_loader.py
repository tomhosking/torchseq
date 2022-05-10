import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchseq.datasets.qa_dataset import QADataset
from torchseq.utils.seed import init_worker
from torchseq.utils.tokenizer import Tokenizer
import torchseq.utils.tokenizer as tokenizer

import logging


class QADataLoader:
    def __init__(self, config, data_path, train_samples=None, dev_samples=None, test_samples=None):
        """
        :param config:
        """
        self.config = config
        self.tokenizer = Tokenizer(config.prepro.get("input_tokenizer", config.prepro.tokenizer), data_path)
        if config.prepro.get("output_tokenizer", None) != config.prepro.get("input_tokenizer", None):
            raise Exception("QADataset doesnt support different input and output tokenizers!")

        self.logger = logging.getLogger("DataLoader")

        tokenizer.DATA_PATH = data_path
        Tokenizer(config.prepro.tokenizer)

        train = QADataset(
            path=os.path.join(data_path, config.training.dataset) + "/"
            if self.config.training.dataset is not None
            else None,
            samples=train_samples,
            config=config,
            tokenizer=self.tokenizer,
            dev=False,
            test=False,
            length_limit=self.config.training.get("truncate_dataset", None),
        )
        valid = QADataset(
            path=os.path.join(data_path, self.config.training.dataset) + "/"
            if self.config.training.dataset is not None
            else None,
            samples=dev_samples,
            config=config,
            tokenizer=self.tokenizer,
            dev=True,
            test=False,
            length_limit=self.config.eval.get("truncate_dataset", None),
        )
        test = QADataset(
            path=os.path.join(data_path, self.config.training.dataset) + "/"
            if self.config.training.dataset is not None
            else None,
            samples=test_samples,
            config=config,
            tokenizer=self.tokenizer,
            dev=False,
            test=True,
            length_limit=self.config.eval.get("truncate_dataset", None),
        )

        self.len_train_data = len(train)
        self.len_valid_data = len(valid)
        self.len_test_data = len(test)

        # TODO: check whether running in silent mode
        self.logger.info(
            "Loaded {:} training and {:} validation examples from {:}".format(
                self.len_train_data, self.len_valid_data, os.path.join(data_path, config.training.dataset)
            )
        )

        self.train_iterations = (
            self.len_train_data + self.config.training.batch_size - 1
        ) // self.config.training.batch_size
        self.valid_iterations = (
            self.len_valid_data + self.config.training.batch_size - 1
        ) // self.config.training.batch_size
        self.test_iterations = (
            self.len_test_data + self.config.training.batch_size - 1
        ) // self.config.training.batch_size

        self.train_loader = DataLoader(
            train,
            batch_size=config.training.batch_size,
            shuffle=self.config.training.data.get("shuffle_data", True),
            num_workers=4,
            collate_fn=QADataset.pad_and_order_sequences(self.tokenizer.pad_id),
            worker_init_fn=init_worker,
        )

        self.valid_loader = DataLoader(
            valid,
            batch_size=config.eval.eval_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=QADataset.pad_and_order_sequences(self.tokenizer.pad_id),
            worker_init_fn=init_worker,
        )

        self.test_loader = DataLoader(
            test,
            batch_size=config.eval.eval_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=QADataset.pad_and_order_sequences(self.tokenizer.pad_id),
            worker_init_fn=init_worker,
        )
