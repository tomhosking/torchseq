import torch
import torch.nn.functional as F
from torch import nn

from torchseq.agents.model_agent import ModelAgent

from torchseq.datasets.lm_dataset import LangmodellingDataset
from torchseq.datasets.lm_loader import LangmodellingDataLoader

from torchseq.models.lm_transformer import TransformerLanguageModel
from torchseq.models.kl_divergence import get_kl
from torchseq.utils.tokenizer import Tokenizer


class LangModelAgent(ModelAgent):
    def __init__(self, config, run_id, output_path, silent=False, training_mode=True):
        super().__init__(config, run_id, output_path, silent, training_mode)

        self.tgt_field = "sent"

        # define data_loader
        if self.config.training.dataset in ["ptb"]:
            self.data_loader = LangmodellingDataLoader(config=config)
            self.src_field = "sent"
        else:
            raise Exception("Unrecognised dataset: {:}".format(config.training.dataset))

        # define loss
        self.loss = nn.CrossEntropyLoss(ignore_index=Tokenizer().pad_id, reduction="none")

        # define model
        self.model = TransformerLanguageModel(self.config, src_field=self.src_field)

        # define optimizer
        self.create_optimizer()

        self.set_device()

        self.create_samplers()
