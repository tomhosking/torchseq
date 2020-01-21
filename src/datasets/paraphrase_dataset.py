
from torch.utils.data import IterableDataset
import torch
import torch.nn.functional as F
from utils.bpe_factory import BPE
import os
from datasets.paraphrase_pair import ParaphrasePair

from itertools import cycle

class ParaphraseDataset(IterableDataset):
    def __init__(self, path, config, dev=False, test=False):
        self.config = config

        self.path = path
        self.variant = 'dev' if dev else 'train'
        
        # TODO: Can we get the length without reading the whole file?
        self.length = 0
        with open(os.path.join(self.path, 'paraphrases.{:}.txt'.format(self.variant))) as f:
            for line in f:
                self.length += 1

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.generator()

    def generator(self):
        worker_info = torch.utils.data.get_worker_info()
        if not worker_info:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        with open(os.path.join(self.path, 'paraphrases.{:}.txt'.format(self.variant))) as f:
            for ix, line in enumerate(f):
                if num_workers > 1 and ix % num_workers != worker_id:
                    continue
                x = line.split('\t')
                sample = {'s1': x[0], 's2': x[1]}
                yield self.to_tensor(sample, tok_window=self.config.prepro.tok_window)

    @staticmethod
    def to_tensor(x, tok_window=64):
        parsed_triple = ParaphrasePair(x['s1'], x['s2'], tok_window=tok_window)

        sample = {'s1': torch.LongTensor(parsed_triple.s1_as_ids()),
                's2': torch.LongTensor(parsed_triple.s2_as_ids()),
                's1_len': torch.LongTensor([len(parsed_triple._s1_doc)]),
                's2_len': torch.LongTensor([len(parsed_triple._s2_doc)]),
                's1_text': x['s1'],
                's2_text': x['s2']
                }

        return sample

    @staticmethod
    def pad_and_order_sequences(batch):
        keys = batch[0].keys()
        max_lens = {k: max(len(x[k]) for x in batch) for k in keys}

        for x in batch:
            for k in keys:
                if k == 'a_pos':
                    x[k] = F.pad(x[k], (0, max_lens[k]-len(x[k])), value=0)
                elif k[-5:] != '_text':
                    x[k] = F.pad(x[k], (0, max_lens[k]-len(x[k])), value=BPE.pad_id)

        tensor_batch = {}
        for k in keys:
            if k[-5:] != '_text':
                tensor_batch[k] = torch.stack([x[k] for x in batch], 0).squeeze(1)
            else:
                tensor_batch[k] = [x[k] for x in batch]

        return tensor_batch
