
from torch.utils.data import Dataset
from datasets.loaders import load_squad_triples
from datasets.cqa_triple import CQATriple
import torch

class SquadDataset(Dataset):
    def __init__(self, path, vocab, dev=False, test=False):
        squad = load_squad_triples(path=path, dev=dev, test=test)
        self.samples =[{'c': x[0], 'q': x[1], 'a': x[2], 'a_pos': x[3]} for x in squad]
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.to_tensor(self.samples[idx])


    def to_tensor(self,x):
        parsed_triple = CQATriple(self.vocab, x['c'], x['a'], x['a_pos'], x['q'])

        sample = {'c': torch.LongTensor(parsed_triple.ctxt_as_ids()),
                    'q': torch.LongTensor(parsed_triple.q_as_ids()),
                    'a': torch.LongTensor(parsed_triple.ans_as_ids()),
                    'a_pos': torch.LongTensor(parsed_triple.ctxt_as_bio()),
                    'c_len': torch.LongTensor([len(parsed_triple._ctxt_doc)]),
                    'a_len': torch.LongTensor([len(parsed_triple._ans_doc)]),
                    'q_len': torch.LongTensor([len(parsed_triple._q_doc)])}

        return sample
