import os
from itertools import cycle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchseq.datasets.paraphrase_pair import ParaphrasePair
from torchseq.utils.tokenizer import Tokenizer


class ParaphraseDataset(Dataset):
    def __init__(self, path, config, dev=False, test=False, repeat=False, length_limit=None):
        self.config = config

        self.repeat = repeat

        self.samples = []

        self.path = path
        self.variant = "dev" if dev else ("test" if test else "train")

        self.exists = True

        self.length = 0

        if test and not os.path.exists(os.path.join(self.path, "paraphrases.{:}.txt".format(self.variant))):
            self.exists = False
        else:
            with open(os.path.join(self.path, "paraphrases.{:}.txt".format(self.variant))) as f:
                for line in f:
                    self.length += 1
                    x = line.strip("\n").split("\t")
                    if len(x) < 2:
                        print(x)
                        print(line)
                        exit()
                    is_para = (True if int(x[2]) > 0 else False) if len(x) > 2 else True
                    sample = {"source": x[0], "target": x[1], "is_para": is_para}
                    self.samples.append(sample)

            if length_limit is not None:
                self.samples = self.samples[:length_limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return ParaphraseDataset.to_tensor(
            self.samples[idx],
            tok_window=self.config.prepro.tok_window,
        )

    # def __iter__(self):
    #     return self.generator()

    # def generator(self):
    #     worker_info = torch.utils.data.get_worker_info()
    #     if not worker_info:
    #         worker_id = 0
    #         num_workers = 1
    #     else:
    #         worker_id = worker_info.id
    #         num_workers = worker_info.num_workers

    #     with open(os.path.join(self.path, "paraphrases.{:}.txt".format(self.variant))) as f:
    #         num_repeats = 0
    #         while self.repeat or num_repeats < 1:
    #             num_repeats += 1
    #             for ix, line in enumerate(f):
    #                 if num_workers > 1 and ix % num_workers != worker_id:
    #                     continue
    #                 x = line.strip("\n").split("\t")
    #                 if len(x) < 2:
    #                     print(x)
    #                     print(line)
    #                     exit()
    #                 is_para = (True if int(x[2]) > 0 else False) if len(x) > 2 else True
    #                 sample = {"source": x[0], "target": x[1], "is_para": is_para}
    #                 yield self.to_tensor(sample, tok_window=self.config.prepro.tok_window)

    @staticmethod
    def to_tensor(x, tok_window=64):
        parsed_triple = ParaphrasePair(
            x["source"],
            x["target"],
            x.get("template", None),
            is_paraphrase=x.get("is_para", True),
            tok_window=tok_window,
        )

        sample = {
            "source": torch.LongTensor(parsed_triple.s1_as_ids()),
            "target": torch.LongTensor(parsed_triple.s2_as_ids()),
            "s1_len": torch.LongTensor([len(parsed_triple._s1_doc)]),
            "s2_len": torch.LongTensor([len(parsed_triple._s2_doc)]),
            "s1_text": x["source"],
            "s2_text": x["target"],
            "is_paraphrase": torch.LongTensor([1 * parsed_triple.is_paraphrase]),
        }

        if "template" in x:
            sample["template"] = torch.LongTensor(parsed_triple.template_as_ids())
            sample["template_len"] = torch.LongTensor([len(parsed_triple._template_doc)])
            sample["template_text"] = x["template"]

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
