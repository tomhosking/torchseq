import torch
import torch.nn.functional as F
from torch import nn

from torchseq.agents.model_agent import ModelAgent


from torchseq.models.lm_transformer import TransformerLanguageModel
from torchseq.models.kl_divergence import get_kl
from torchseq.utils.tokenizer import Tokenizer


class LangModelAgent(ModelAgent):
    def __init__(
        self,
        config,
        run_id,
        output_path,
        silent=False,
        training_mode=True,
        verbose=True,
        cache_root=None,
    ):
        super().__init__(config, run_id, output_path, silent, training_mode, verbose, cache_root)

        self.tgt_field = "sent"

        self.src_field = "sent"

        # define loss
        self.loss = nn.CrossEntropyLoss(ignore_index=Tokenizer().pad_id, reduction="none")

        # define model
        self.model = TransformerLanguageModel(self.config, src_field=self.src_field)

        # define optimizer
        if training_mode:
            self.create_optimizer()

        self.set_device()

        self.create_samplers()
