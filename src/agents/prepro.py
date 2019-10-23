import os, json
from datasets.squad_dataset import SquadDataset
import torch

class PreprocessorAgent():
    def __init__(self, config):
        self.train = SquadDataset(os.path.join(config.data_path, 'squad/'), dev=False, test=False)
        self.valid = SquadDataset(os.path.join(config.data_path, 'squad/'), dev=True, test=False)

        self.output_path = os.path.join(config.data_path, 'processed/')
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def run(self):


        train_processed = []
        for example in self.train:
            train_processed.append(example)

        
        torch.save(train_processed, os.path.join(self.output_path, 'train_processed.pt'))


        valid_processed = []
        for example in self.valid:
            valid_processed.append(example)

        torch.save(valid_processed, os.path.join(self.output_path, 'valid_processed.pt'))
