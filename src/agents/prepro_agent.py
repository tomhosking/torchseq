import json
import os

import torch

from agents.base import BaseAgent
from datasets.paraphrase_dataset import ParaphraseDataset
from datasets.squad_dataset import SquadDataset


class PreprocessorAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        if self.config.training.dataset == "squad":
            self.train = SquadDataset(os.path.join(config.env.data_path, "squad/"), config, dev=False, test=False)
            self.valid = SquadDataset(os.path.join(config.env.data_path, "squad/"), config, dev=True, test=False)
            self.test = SquadDataset(os.path.join(config.env.data_path, "squad/"), config, dev=False, test=True)

        elif self.config.training.dataset == "paranmt":
            self.train = ParaphraseDataset(
                os.path.join(config.env.data_path, "paranmt/"), config, dev=False, test=False
            )
            self.valid = ParaphraseDataset(
                os.path.join(config.env.data_path, "paranmt/"), config, dev=True, test=False
            )
            self.test = (
                []
            )  # SquadDataset(os.path.join(config.env.data_path, 'paranmt/'), config, dev=False, test=True)

        else:
            raise Exception("Unrecognised dataset passed to preprocessor: {:}".format(self.config.training.dataset))

        self.output_path = os.path.join(config.env.data_path, "processed", self.config.training.dataset)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def run(self):

        test_processed = []
        for example in self.test:
            test_processed.append(example)

        torch.save(test_processed, os.path.join(self.output_path, "test_processed.pt"))

        train_processed = []
        for example in self.train:
            train_processed.append(example)

        torch.save(train_processed, os.path.join(self.output_path, "train_processed.pt"))

        valid_processed = []
        for example in self.valid:
            valid_processed.append(example)

        torch.save(valid_processed, os.path.join(self.output_path, "valid_processed.pt"))
