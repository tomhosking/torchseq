import os
import json
import jsonlines

# from itertools import cycle

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
import numpy as np

from torchseq.utils.tokenizer import FAIRSEQ_LANGUAGE_CODES


class JsonDataset(Dataset):
    def __init__(
        self,
        config,
        input_tokenizer,
        output_tokenizer,
        path=None,
        samples=None,
        dev=False,
        test=False,
        repeat=False,
        length_limit=None,
        repeat_samples=None,
    ):
        self.config = config
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

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

    def __getitem__(self, idx):
        return JsonDataset.to_tensor(
            self.samples[idx],
            self.fields,
            self.input_tokenizer,
            self.output_tokenizer,
            tok_window=self.config.prepro.tok_window,
            include_lang_codes=self.config.prepro.data.get("include_lang_codes", False),
            mask_prob=self.config.prepro.data.get("token_mask_prob", 0),
        )

    @staticmethod
    def to_tensor(
        obj, fields, input_tokenizer, output_tokenizer, tok_window=64, include_lang_codes=False, mask_prob=0.0
    ):

        src_lang = obj.get("src_lang", "en_XX")
        tgt_lang = obj.get("tgt_lang", "en_XX")

        if include_lang_codes:
            src_lang_token = FAIRSEQ_LANGUAGE_CODES[src_lang]
            tgt_lang_token = FAIRSEQ_LANGUAGE_CODES[tgt_lang]

        transformed = {}

        for f in fields:
            transformed[f["to"]] = JsonDataset._transform(obj[f["from"]], f.get("type", "copy"))

        batch = {k + "_text": v for k, v in transformed.items()}
        for f in fields:
            # HACK: this should be in a config somewhere...
            if include_lang_codes and f["to"] == "source":
                lang_tok = [src_lang_token]
            elif include_lang_codes and f["to"] == "target":
                lang_tok = [tgt_lang_token]
            else:
                lang_tok = []

            field_values = JsonDataset._tokenize_if_string(
                transformed[f["to"]],
                (output_tokenizer if f.get("tokenizer", "input") == "output" else input_tokenizer),
                tok_window,
            )

            if include_lang_codes:
                batch[f["to"]] = torch.LongTensor(lang_tok + field_values)
                batch[f["to"] + "_len"] = torch.LongTensor([len(batch[f["to"]]) + len(lang_tok)])
            elif f["to"][0] != "_" or f.get("to_tensor", True) is False:
                if isinstance(field_values, torch.Tensor):
                    if torch.is_floating_point(field_values):
                        batch[f["to"]] = torch.FloatTensor(field_values.tolist())
                    else:
                        batch[f["to"]] = torch.LongTensor(field_values.tolist())
                else:
                    batch[f["to"]] = torch.LongTensor(field_values)
                batch[f["to"] + "_len"] = torch.LongTensor([len(batch[f["to"]])])
            else:
                batch[f["to"]] = field_values

            # HACK: hard coded field name!
            if f["to"] == "source" and mask_prob > 0:

                probs = torch.rand(*batch[f["to"]].shape, device=batch[f["to"]].device)
                mask = torch.logical_and(
                    torch.logical_and(probs < mask_prob, batch[f["to"]] > 3), batch[f["to"]] < 250000
                )

                batch[f["to"]] = torch.where(mask, input_tokenizer.mask_id, batch[f["to"]])

        if include_lang_codes:
            batch["src_lang"] = torch.LongTensor([src_lang_token])
            batch["tgt_lang"] = torch.LongTensor([tgt_lang_token])

        return batch

    @staticmethod
    def _transform(value, transform_type="copy"):
        if transform_type == "sample":
            return np.random.choice(value)
        elif transform_type == "copy":
            return value
        else:
            raise Exception(f"Unrecognised transform type in JsonDataset: {transform_type}")

    @staticmethod
    def _tokenize_if_string(value, tokenizer, tok_window):
        if isinstance(value, str):

            _doc = tokenizer.tokenise(value)

            if len(_doc) > tok_window:
                _doc = _doc[:tok_window]

            return [tok["id"] for tok in _doc]
        else:
            return value

    @staticmethod
    def pad_and_order_sequences(pad_id):
        def _pad_and_order_sequences(batch):
            keys = batch[0].keys()
            max_lens = {k: max(len(x[k]) for x in batch) for k in keys}

            for x in batch:
                for k in keys:
                    if k[0] == "_":
                        continue
                    if k == "a_pos":
                        x[k] = F.pad(x[k], (0, max_lens[k] - len(x[k])), value=0)
                    elif k[-5:] != "_text":
                        x[k] = F.pad(x[k], (0, max_lens[k] - len(x[k])), value=pad_id)

            tensor_batch = {}
            for k in keys:
                if k[-5:] != "_text" and k[0] != "_":
                    tensor_batch[k] = torch.stack([x[k] for x in batch], 0)
                    if k[-4:] == "_len":
                        tensor_batch[k] = tensor_batch[k].squeeze(dim=1)
                else:
                    tensor_batch[k] = [x[k] for x in batch]

            return tensor_batch

        return _pad_and_order_sequences
