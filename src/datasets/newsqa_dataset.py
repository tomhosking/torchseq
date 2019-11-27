
from torch.utils.data import Dataset

from datasets.cqa_triple import CQATriple
import torch

import os,json

class NewsqaDataset(Dataset):
    def __init__(self, path, config, dev=False, test=False):
        self.config = config

        split_as_str = 'test' if test else 'dev' if dev else 'train'

        with open(os.path.join(path, 'combined-newsqa-data-v1.json')) as f:
            all_data = json.load(f)
        
        self.samples = []

        for story in all_data['data']:
            
            if story['type'] == split_as_str:
                for question in story['questions']:
                    if not ('badQuestion' in question['consensus'] and question['consensus']['badQuestion']) and not ('noAnswer' in question['consensus'] and question['consensus']['noAnswer']):
                        a_text = story['text'][question['consensus']['s']:question['consensus']['e']].rstrip()
                        sample = {
                            'c': story['text'].replace('\n',' '),
                            'q': question['q'],
                            'a': a_text,
                            'a_pos': question['consensus']['s']
                        }
                        # print(question)
                        # print(sample)
                        # exit()
                        self.samples.append(sample)
        # self.samples =[{'c': x[0], 'q': x[1], 'a': x[2], 'a_pos': x[3]} for x in squad if (len(x[1]) < 200 and len(x[0]) < 3500 and x[1] != "I couldn't could up with another question. But i need to fill this space because I can't submit the hit. ") or test is True]
        


        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.to_tensor(self.samples[idx])


    def to_tensor(self,x):
        parsed_triple = CQATriple(x['c'], x['a'], x['a_pos'], x['q'])

        sample = {'c': torch.LongTensor(parsed_triple.ctxt_as_ids()),
                    'q': torch.LongTensor(parsed_triple.q_as_ids()),
                    'a': torch.LongTensor(parsed_triple.ans_as_ids()),
                    'a_pos': torch.LongTensor(parsed_triple.ctxt_as_bio()),
                    'c_len': torch.LongTensor([len(parsed_triple._ctxt_doc)]),
                    'a_len': torch.LongTensor([len(parsed_triple._ans_doc)]),
                    'q_len': torch.LongTensor([len(parsed_triple._q_doc)])}

        return sample
