import os
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchseq.datasets.json_dataset import JsonDataset
from torchseq.utils.seed import init_worker
from torchseq.utils.tokenizer import Tokenizer


class JsonDataLoader:
    def __init__(self, config, train_samples=None, dev_samples=None, test_samples=None):
        """
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("DataLoader")

        self._train = JsonDataset(
            config=config,
            path=os.path.join(config.env.data_path, self.config.json_dataset.path)
            if self.config.json_dataset.path is not None
            else None,
            samples=train_samples,
            dev=False,
            test=False,
            repeat=(self.config.training.data.get("epoch_steps", 0) > 0),
            length_limit=self.config.training.get("truncate_dataset", None),
            repeat_samples=self.config.training.get("repeat_samples", None),
        )
        self._valid = JsonDataset(
            config=config,
            path=os.path.join(config.env.data_path, self.config.json_dataset.path)
            if self.config.json_dataset.path is not None
            else None,
            samples=dev_samples,
            dev=True,
            test=False,
            length_limit=self.config.eval.get("truncate_dataset", None),
            repeat_samples=self.config.eval.get("repeat_samples", None),
        )
        self._test = JsonDataset(
            config=config,
            path=os.path.join(config.env.data_path, self.config.json_dataset.path)
            if self.config.json_dataset.path is not None
            else None,
            samples=test_samples,
            dev=False,
            test=True,
            length_limit=self.config.eval.get("truncate_dataset", None),
            repeat_samples=self.config.eval.get("repeat_samples", None),
        )

        self.len_train_data = len(self._train)
        self.len_valid_data = len(self._valid)
        # self.len_test_data = len(test)

        # TODO: check whether running in silent mode
        # self.logger.info(
        #     "Loaded {:} training and {:} validation examples from {:}".format(
        #         self.len_train_data,
        #         self.len_valid_data,
        #         os.path.join(config.env.data_path, self.config.json_dataset.path) if ,
        #     )
        # )

        # self.train_iterations = (self.len_train_data + self.config.training.batch_size - 1) // self.config.training.batch_size
        # self.valid_iterations = (self.len_valid_data + self.config.training.batch_size - 1) // self.config.training.batch_size
        # self.test_iterations = (self.len_test_data + self.config.training.batch_size - 1) // self.config.training.batch_size

        if self._train.exists:
            self.train_loader = DataLoader(
                self._train,
                batch_size=config.training.batch_size,
                shuffle=self.config.training.data.get("shuffle_data", True),
                num_workers=2,
                collate_fn=JsonDataset.pad_and_order_sequences,
                worker_init_fn=init_worker,
            )

        if self._valid.exists:
            self.valid_loader = DataLoader(
                self._valid,
                batch_size=config.eval.eval_batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=JsonDataset.pad_and_order_sequences,
                worker_init_fn=init_worker,
            )
        if self._test.exists:
            self.test_loader = DataLoader(
                self._test,
                batch_size=config.eval.eval_batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=JsonDataset.pad_and_order_sequences,
                worker_init_fn=init_worker,
            )
