
import torch
from torch import nn

from agents.model_agent import ModelAgent

from models.para_transformer import TransformerParaphraseModel

from datasets.paraphrase_loader import ParaphraseDataLoader

class ParaphraseAgent(ModelAgent):

    def __init__(self, config, run_id, silent=False):
        super().__init__(config, run_id, silent)

        # define data_loader
        if self.config.training.use_preprocessed_data:
            self.data_loader = PreprocessedDataLoader(config=config)
        else:
            if self.config.training.dataset == 'paranmt':
                self.data_loader = ParaphraseDataLoader(config=config)
            else:
                raise Exception("Unrecognised dataset: {:}".format(config.training.dataset))

        # define model
        self.model = TransformerParaphraseModel(self.config)


        # define loss
        self.loss = nn.CrossEntropyLoss(ignore_index=BPE.pad_id, reduction='none')



        # define optimizer
        self.create_optimizer()

        self.set_device()

        self.model.device = self.device

        self.create_samplers()

        