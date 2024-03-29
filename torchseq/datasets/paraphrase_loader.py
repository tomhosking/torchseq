import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchseq.datasets.paraphrase_dataset import ParaphraseDataset
from torchseq.utils.seed import init_worker
from torchseq.utils.tokenizer import Tokenizer
import torchseq.utils.tokenizer as tokenizer
import logging


class ParaphraseDataLoader:
    def __init__(self, config, data_path):
        """
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("DataLoader")

        raise Exception("Paraphrase DataLoader is deprecated! Use JsonDataset instead")

        tokenizer.DATA_PATH = data_path
        Tokenizer(config.prepro.tokenizer)

        train = ParaphraseDataset(
            os.path.join(data_path, self.config.training.dataset),
            config=config,
            dev=False,
            test=False,
            repeat=(self.config.training.data.get("epoch_steps", 0) > 0),
            length_limit=self.config.training.get("truncate_dataset", None),
        )
        valid = ParaphraseDataset(
            os.path.join(data_path, self.config.training.dataset),
            config=config,
            dev=True,
            test=False,
            length_limit=self.config.eval.get("truncate_dataset", None),
        )
        test = ParaphraseDataset(
            os.path.join(data_path, self.config.training.dataset),
            config=config,
            dev=False,
            test=True,
            length_limit=self.config.eval.get("truncate_dataset", None),
        )

        self.len_train_data = len(train)
        self.len_valid_data = len(valid)
        # self.len_test_data = len(test)

        # TODO: check whether running in silent mode
        self.logger.info(
            "Loaded {:} training and {:} validation examples from {:}".format(
                self.len_train_data,
                self.len_valid_data,
                os.path.join(data_path, self.config.training.dataset),
            )
        )

        # self.train_iterations = (self.len_train_data + self.config.training.batch_size - 1) // self.config.training.batch_size
        # self.valid_iterations = (self.len_valid_data + self.config.training.batch_size - 1) // self.config.training.batch_size
        # self.test_iterations = (self.len_test_data + self.config.training.batch_size - 1) // self.config.training.batch_size

        self.train_loader = DataLoader(
            train,
            batch_size=config.training.batch_size,
            # shuffle=self.config.training.data.get("shuffle_data", True),
            num_workers=0,
            collate_fn=ParaphraseDataset.pad_and_order_sequences,
            worker_init_fn=init_worker,
        )

        self.valid_loader = DataLoader(
            valid,
            batch_size=config.eval.eval_batch_size,
            # shuffle=False,
            num_workers=0,
            collate_fn=ParaphraseDataset.pad_and_order_sequences,
            worker_init_fn=init_worker,
        )
        if test.exists:
            self.test_loader = DataLoader(
                test,
                batch_size=config.eval.eval_batch_size,
                # shuffle=False,
                num_workers=0,
                collate_fn=ParaphraseDataset.pad_and_order_sequences,
                worker_init_fn=init_worker,
            )
