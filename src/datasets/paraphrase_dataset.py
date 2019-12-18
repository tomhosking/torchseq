
from torch.utils.data import Dataset
import torch
import os
from datasets.paraphrase_pair import ParaphrasePair

class ParaphraseDataset(Dataset):
    def __init__(self, path, config, dev=False, test=False):
        self.config = config
        
        with open(os.path.join(path, 'para-nmt-50m.txt')) as f:
            paraphrases = f.readlines()

        self.samples =[{'s1': x[0], 's2': x[1]} for x in paraphrases if float(x[2]) < 0.8]
        


        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.to_tensor(self.samples[idx])


    def to_tensor(self,x):
        parsed_triple = ParaphrasePair(x['s1'], x['s2'])

        sample = {'s1': torch.LongTensor(parsed_triple.s1_as_ids()),
                's2': torch.LongTensor(parsed_triple.s2_as_ids()),
                's1_len': torch.LongTensor([len(parsed_triple._s1_doc)]),
                's2_len': torch.LongTensor([len(parsed_triple._s2_doc)])}

        return sample
