
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.paraphrase_dataset import ParaphraseDataset
from utils.bpe_factory import BPE
from utils.seed import init_worker

class ParaphraseDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        train = ParaphraseDataset(os.path.join(config.env.data_path, self.config.training.dataset), config=config, dev=False, test=False)
        valid = ParaphraseDataset(os.path.join(config.env.data_path, self.config.training.dataset), config=config, dev=True, test=False)
        # test = ParaphraseDataset(os.path.join(config.env.data_path, 'squad'), config=config, dev=False, test=True)

        self.len_train_data = len(train)
        self.len_valid_data = len(valid)
        # self.len_test_data = len(test)

        print("Loaded {:} training and {:} validation examples from {:}".format(self.len_train_data, self.len_valid_data, os.path.join(config.env.data_path, self.config.training.dataset)))

        # self.train_iterations = (self.len_train_data + self.config.training.batch_size - 1) // self.config.training.batch_size
        # self.valid_iterations = (self.len_valid_data + self.config.training.batch_size - 1) // self.config.training.batch_size
        # self.test_iterations = (self.len_test_data + self.config.training.batch_size - 1) // self.config.training.batch_size


        self.train_loader = DataLoader(train,
                                        batch_size=config.training.batch_size, 
                                        shuffle=False, 
                                        num_workers=1, 
                                        collate_fn=self.pad_and_order_sequences, 
                                        worker_init_fn=init_worker)

        self.valid_loader = DataLoader(valid, 
                                        batch_size=config.eval.eval_batch_size, 
                                        shuffle=False, 
                                        num_workers=1,
                                        collate_fn=self.pad_and_order_sequences, 
                                        worker_init_fn=init_worker)
                                    
        # self.test_loader = DataLoader(test, 
        #                                 batch_size=config.eval.eval_batch_size, 
        #                                 shuffle=False, 
        #                                 num_workers=6,
        #                                 collate_fn=self.pad_and_order_sequences, 
        #                                 worker_init_fn=init_worker)

    def pad_and_order_sequences(self, batch):
        keys = batch[0].keys()
        max_lens = {k: max(len(x[k]) for x in batch) for k in keys}

        for x in batch:
            for k in keys:
                if k == 'a_pos':
                    x[k] = F.pad(x[k], (0, max_lens[k]-len(x[k])), value=0)
                else:
                    x[k] = F.pad(x[k], (0, max_lens[k]-len(x[k])), value=BPE.pad_id)

        tensor_batch = {}
        for k in keys:
            tensor_batch[k] = torch.stack([x[k] for x in batch], 0).squeeze(1)

        return tensor_batch
