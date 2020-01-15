
from torch.utils.data import IterableDataset
import torch
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
                yield self.to_tensor(sample)


    def to_tensor(self,x):
        parsed_triple = ParaphrasePair(x['s1'], x['s2'], config=self.config)

        sample = {'s1': torch.LongTensor(parsed_triple.s1_as_ids()),
                's2': torch.LongTensor(parsed_triple.s2_as_ids()),
                's1_len': torch.LongTensor([len(parsed_triple._s1_doc)]),
                's2_len': torch.LongTensor([len(parsed_triple._s2_doc)])}

        return sample
