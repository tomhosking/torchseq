
from torch.utils.data import Dataset
from datasets.loaders import load_squad_triples
from datasets.cqa_triple import CQATriple
import torch
import torch.nn.functional as F


from utils.bpe_factory import BPE

import json, os

class SquadDataset(Dataset):
    def __init__(self, path, config, dev=False, test=False):
        self.config = config

        if config.training.dataset == 'squad':
            squad = load_squad_triples(path=path, dev=dev, test=test)
            
            self.samples =[{'c': x[0], 'q': x[1], 'a': x[2], 'a_pos': x[3]} for x in squad if (len(x[1]) < 200 and len(x[0]) < 3500 and x[1] != "I couldn't could up with another question. But i need to fill this space because I can't submit the hit. ") or test is True]
        else:
            self.variant = 'dev' if dev else ('test' if test else 'train')
            with open(os.path.join(path, config.training.dataset, 'questions.{:}.json'.format(self.variant))) as f:
                self.samples = json.load(f)
        


        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return SquadDataset.to_tensor(self.samples[idx],
            sent_window=self.config.prepro.sent_window,
            tok_window=self.config.prepro.tok_window,
            o_tag=2 if self.config.prepro.bio_tagging else 1,
            concat_ctxt_ans=self.config.prepro.concat_ctxt_ans)

    @staticmethod
    def to_tensor(x,
            sent_window=0,
            tok_window=300,
            o_tag=2,
            concat_ctxt_ans=False):

        parsed_triple = CQATriple(x['c'], x['a'], x['a_pos'], x['q'],
            sent_window=sent_window,
            tok_window=tok_window,
            o_tag=o_tag)

        if concat_ctxt_ans:
            sample = {'c':  torch.LongTensor(parsed_triple.ctxt_as_ids() + [BPE.eos_id] +  parsed_triple.ans_as_ids()),
                    'q': torch.LongTensor(parsed_triple.q_as_ids()),
                    'a': torch.LongTensor(parsed_triple.ans_as_ids()),
                    'a_pos': torch.LongTensor([0 for i in range(len(parsed_triple._ctxt_doc))] + [0] + [1 for i in range(len(parsed_triple._ans_doc))]),
                    'c_len': torch.LongTensor([len(parsed_triple._ctxt_doc) + len(parsed_triple._ans_doc) + 1]),
                    'a_len': torch.LongTensor([len(parsed_triple._ans_doc)]),
                    'q_len': torch.LongTensor([len(parsed_triple._q_doc)]),
                    'c_text': x['c'],
                    'a_text': x['a'],
                    'q_text': x['q']
                    }
        else:
            sample = {'c': torch.LongTensor(parsed_triple.ctxt_as_ids()),
                    'q': torch.LongTensor(parsed_triple.q_as_ids()),
                    'a': torch.LongTensor(parsed_triple.ans_as_ids()),
                    'a_pos': torch.LongTensor(parsed_triple.ctxt_as_bio()),
                    'c_len': torch.LongTensor([len(parsed_triple._ctxt_doc)]),
                    'a_len': torch.LongTensor([len(parsed_triple._ans_doc)]),
                    'q_len': torch.LongTensor([len(parsed_triple._q_doc)]),
                    'c_text': x['c'],
                    'a_text': x['a'],
                    'q_text': x['q']
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
