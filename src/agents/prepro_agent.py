import os, json
from datasets.squad_dataset import SquadDataset
import torch
from agents.base import BaseAgent

class PreprocessorAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        
        self.train = SquadDataset(os.path.join(config.env.data_path, 'squad/'), config,  dev=False, test=False)
        self.valid = SquadDataset(os.path.join(config.env.data_path, 'squad/'), config, dev=True, test=False)
        self.valid = SquadDataset(os.path.join(config.env.data_path, 'squad/'), config, dev=False, test=True)

        self.output_path = os.path.join(config.env.data_path, 'processed/')
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def run(self):

        test_processed = []
        for example in self.valid:
            test_processed.append(example)

        torch.save(test_processed, os.path.join(self.output_path, 'test_processed.pt'))

        train_processed = []
        for example in self.train:
            train_processed.append(example)

        
        torch.save(train_processed, os.path.join(self.output_path, 'train_processed.pt'))


        valid_processed = []
        for example in self.valid:
            valid_processed.append(example)

        torch.save(valid_processed, os.path.join(self.output_path, 'valid_processed.pt'))


