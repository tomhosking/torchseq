import json
import os
import jsonlines

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchseq.datasets.cqa_triple import CQATriple
from torchseq.datasets.loaders import load_squad_triples
from torchseq.utils.tokenizer import Tokenizer


class SquadDataset(Dataset):
    def __init__(self, path, config, dev=False, test=False):
        self.config = config

        if config.training.dataset == "squad":
            squad = load_squad_triples(path=path, dev=dev, test=test)

            self.samples = [
                {"c": x[0], "q": x[1], "a": x[2], "a_pos": x[3]}
                for x in squad
                if (
                    len(x[1]) < 200
                    and len(x[0]) < 3500
                    and x[1]
                    != "I couldn't could up with another question. But i need to fill this space because I can't submit the hit. "
                )
                or test is True
            ]
        else:
            self.variant = "dev" if dev else ("test" if test else "train")
            file_path = os.path.join(path, "questions.{:}.json".format(self.variant))

            # JSON format
            if os.path.exists(file_path):
                with open(file_path) as f:
                    self.samples = json.load(f)

            # JSONlines format
            elif os.path.exists(file_path + "l"):
                with open(file_path + "l") as f:
                    reader = jsonlines.Reader(f)
                    self.samples = [
                        x for x in reader
                    ]  # TODO: this will explode for large files - would be nice to auto detect jsonlines files and switch to a streaming dataset

                    reader.close()
            else:
                raise Exception("Couldn't find dataset file! {:}".format(file_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return SquadDataset.to_tensor(
            self.samples[idx],
            sent_window=self.config.prepro.sent_window,
            tok_window=self.config.prepro.tok_window,
            o_tag=2 if self.config.prepro.bio_tagging else 1,
            concat_ctxt_ans=self.config.prepro.concat_ctxt_ans,
            roberta_style_encoding=self.config.prepro.data.get("roberta_style_encoding", False),
        )

    @staticmethod
    def to_tensor(x, sent_window=0, tok_window=300, o_tag=2, concat_ctxt_ans=False, roberta_style_encoding=False):

        parsed_triple = CQATriple(
            x["c"], x["a"], x["a_pos"], x["q"], sent_window=sent_window, tok_window=tok_window, o_tag=o_tag
        )

        if concat_ctxt_ans:
            if roberta_style_encoding:
                # Roberta sequence pairs look like <s>A</s></s>B</s> for no obvious reason
                ctxt = torch.LongTensor(
                    parsed_triple.ctxt_as_ids() + [Tokenizer().eos_id] * 2 + parsed_triple.ans_as_ids()[1:]
                )
            else:
                ctxt = torch.LongTensor(
                    parsed_triple.ans_as_ids() + [Tokenizer().eos_id] + parsed_triple.ctxt_as_ids()
                )
            sample = {
                "c": ctxt,
                "q": torch.LongTensor(parsed_triple.q_as_ids()),
                "a": torch.LongTensor(parsed_triple.ans_as_ids()),
                "a_pos": torch.LongTensor(
                    [0 for i in range(len(parsed_triple._ans_doc))]
                    + [1]
                    + [1 for i in range(len(parsed_triple._ctxt_doc))]
                ),
                "c_len": torch.LongTensor([len(parsed_triple._ctxt_doc) + len(parsed_triple._ans_doc) + 1]),
                "a_len": torch.LongTensor([len(parsed_triple._ans_doc)]),
                "q_len": torch.LongTensor([len(parsed_triple._q_doc)]),
                "c_text": x["c"],
                "a_text": x["a"],
                "q_text": x["q"],
            }
        else:
            sample = {
                "c": torch.LongTensor(parsed_triple.ctxt_as_ids()),
                "q": torch.LongTensor(parsed_triple.q_as_ids()),
                "a": torch.LongTensor(parsed_triple.ans_as_ids()),
                "a_pos": torch.LongTensor(parsed_triple.ctxt_as_bio()),
                "c_len": torch.LongTensor([len(parsed_triple._ctxt_doc)]),
                "a_len": torch.LongTensor([len(parsed_triple._ans_doc)]),
                "q_len": torch.LongTensor([len(parsed_triple._q_doc)]),
                "c_text": x["c"],
                "a_text": x["a"],
                "q_text": x["q"],
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
