import os
import json
import jsonlines

# from itertools import cycle

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
import numpy as np

from torchseq.utils.tokenizer import Tokenizer, FAIRSEQ_LANGUAGE_CODES


# class LangmodellingDataset(IterableDataset):
class JsonDataset(Dataset):
    def __init__(
        self,
        config,
        path=None,
        samples=None,
        dev=False,
        test=False,
        repeat=False,
        length_limit=None,
        repeat_samples=None,
    ):
        self.config = config

        self.repeat = repeat

        self.path = path

        split = "dev" if dev else ("test" if test else "train")
        self.variant = self.config.json_dataset.get("filename", "{split}").format(split=split)

        self.exists = True

        self.fields = self.config.json_dataset.data["field_map"]

        self.length = 0

        if samples is not None:
            self.samples = samples
        elif path is None or not (
            os.path.exists(os.path.join(self.path, "{:}.json".format(self.variant)))
            or os.path.exists(os.path.join(self.path, "{:}.jsonl".format(self.variant)))
        ):
            self.exists = False
            self.samples = []
        else:

            # TODO: Can we get the length without reading the whole file?
            if os.path.exists(os.path.join(self.path, "{:}.json".format(self.variant))):
                with open(os.path.join(self.path, "{:}.json".format(self.variant))) as f:
                    self.samples = json.load(f)
            elif os.path.exists(os.path.join(self.path, "{:}.jsonl".format(self.variant))):
                with jsonlines.open(os.path.join(self.path, "{:}.jsonl".format(self.variant))) as f:
                    # TODO: maybe jsonl files should be read as a stream instead of all in one...
                    self.samples = [x for x in f]
            else:
                fpath = os.path.join(self.path, "{:}.json(l)".format(self.variant))
                raise Exception("Could not find dataset file! {:}".format(fpath))

            if length_limit is not None:
                self.samples = self.samples[:length_limit]

            if repeat_samples is not None:
                self.samples = [x for x in self.samples for _ in range(repeat_samples)]

    def __len__(self):
        return len(self.samples)

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

    #     with open(os.path.join(self.path, "sentences.{:}.txt".format(self.variant))) as f:
    #         num_repeats = 0
    #         while self.repeat or num_repeats < 1:
    #             num_repeats += 1
    #             for ix, line in enumerate(f):
    #                 if num_workers > 1 and ix % num_workers != worker_id:
    #                     continue
    #                 x = line.strip("\n")

    #                 sample = {"sent": x}
    #                 yield self.to_tensor(sample, tok_window=self.config.prepro.tok_window)

    def __getitem__(self, idx):
        return JsonDataset.to_tensor(
            self.samples[idx],
            self.fields,
            tok_window=self.config.prepro.tok_window,
            include_lang_codes=self.config.prepro.data.get("include_lang_codes", False),
        )

    @staticmethod
    def to_tensor(obj, fields, tok_window=64, include_lang_codes=False):

        src_lang = obj.get("src_lang", "en_XX")
        tgt_lang = obj.get("tgt_lang", "en_XX")

        if include_lang_codes:
            src_lang_token = FAIRSEQ_LANGUAGE_CODES[src_lang]
            tgt_lang_token = FAIRSEQ_LANGUAGE_CODES[tgt_lang]

        sample = {}

        for f in fields:
            sample[f["to"]] = JsonDataset._transform(obj[f["from"]], f.get("type", "copy"))

        parsed = JsonDataInstance(sample, [f["to"] for f in fields])

        sample = {k + "_text": v for k, v in sample.items()}
        for f in fields:
            # HACK: this should be in a config somewhere...
            if include_lang_codes and f["to"] == "s1":
                lang_tok = [src_lang_token]
            elif include_lang_codes and f["to"] == "s2":
                lang_tok = [tgt_lang_token]
            else:
                lang_tok = []

            sample[f["to"]] = torch.LongTensor(lang_tok + parsed.field_as_ids(f["to"]))
            sample[f["to"] + "_len"] = torch.LongTensor([len(sample[f["to"]]) + len(lang_tok)])

        if include_lang_codes:
            sample["src_lang"] = torch.LongTensor([src_lang_token])
            sample["tgt_lang"] = torch.LongTensor([tgt_lang_token])

        return sample

    @staticmethod
    def _transform(value, transform_type="copy"):
        if transform_type == "sample":
            return np.random.choice(value)
        elif transform_type == "copy":
            return value
        else:
            raise Exception(f"Unrecognised transform type in JsonDataset: {transform_type}")

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


class JsonDataInstance:
    def __init__(self, sample, fields, tok_window=256):

        self._ids = {}
        for f in fields:
            if isinstance(sample[f], str):

                _doc = Tokenizer().tokenise(sample[f])

                if len(_doc) > tok_window:
                    _doc = _doc[:tok_window]

                self._ids[f] = [tok["id"] for tok in _doc]
            else:
                self._ids[f] = sample[f]

    def field_as_ids(self, field):

        return self._ids[field]
