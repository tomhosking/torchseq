import os
from itertools import cycle

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset


from torchseq.utils.tokenizer import Tokenizer


class LangmodellingDataset(IterableDataset):
    def __init__(self, path, config, dev=False, test=False, repeat=False):
        self.config = config

        self.repeat = repeat

        self.path = path
        self.variant = "dev" if dev else ("test" if test else "train")

        self.exists = True

        self.length = 0

        if test and not os.path.exists(os.path.join(self.path, "samples.{:}.txt".format(self.variant))):
            self.exists = False
        else:

            # TODO: Can we get the length without reading the whole file?
            with open(os.path.join(self.path, "samples.{:}.txt".format(self.variant))) as f:
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

        with open(os.path.join(self.path, "paraphrases.{:}.txt".format(self.variant))) as f:
            num_repeats = 0
            while self.repeat or num_repeats < 1:
                num_repeats += 1
                for ix, line in enumerate(f):
                    if num_workers > 1 and ix % num_workers != worker_id:
                        continue
                    x = line.strip("\n")

                    sample = {"sent": x}
                    yield self.to_tensor(sample, tok_window=self.config.prepro.tok_window)

    @staticmethod
    def to_tensor(x, tok_window=64):
        parsed = LangmodellingInstance(x["sent"])

        sample = {
            "sent": torch.LongTensor(parsed.sent_as_ids()),
            "sent_len": torch.LongTensor([len(parsed._doc)]),
            "sent_text": x["sent"],
        }

        return sample

    @staticmethod
    def pad_and_order_sequences(batch):
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
            if k[-5:] != "_text":
                tensor_batch[k] = torch.stack([x[k] for x in batch], 0).squeeze(1)
            else:
                tensor_batch[k] = [x[k] for x in batch]

        return tensor_batch


class LangmodellingInstance:
    def __init__(self, sent_text, tok_window=256):

        self._doc = Tokenizer().tokenise(sent_text)

        if len(self._doc) > tok_window:
            self._doc = self._doc[:tok_window]

    def sent_as_ids(self):
        id_list = [tok["id"] for tok in self._doc]
        return id_list
