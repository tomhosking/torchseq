import torch
import torch.nn.functional as F
from torch import nn

from torchseq.agents.model_agent import ModelAgent


from torchseq.models.lm_transformer import TransformerLanguageModel
from torchseq.models.kl_divergence import gaussian_kl
from torchseq.utils.tokenizer import Tokenizer


class LangModelAgent(ModelAgent):
    def __init__(
        self,
        config,
        run_id,
        output_path,
        data_path,
        silent=False,
        training_mode=True,
        verbose=True,
        cache_root=None,
        use_cuda=True,
    ):
        super().__init__(config, run_id, output_path, data_path, silent, training_mode, verbose, cache_root)

        self.tgt_field = "sent"

        self.src_field = "sent"

        # define loss
        self.loss = nn.CrossEntropyLoss(
            ignore_index=self.output_tokenizer.pad_id,
            reduction="none",
            label_smoothing=self.config.training.get("label_smoothing", 0.0),
        )

        # define model
        self.model = TransformerLanguageModel(self.config, src_field=self.src_field)

        # define optimizer
        if training_mode:
            self.create_optimizer()

        self.set_device(use_cuda)

        self.create_samplers()
