from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from torchseq.agents.model_agent import ModelAgent

from torchseq.models.retrieval_model import RetrievalModel
from torchseq.utils.loss_dropper import LossDropper
from torchseq.models.contrastive_loss import ContrastiveLoss

from torchseq.utils.logging import Logger


class RetrievalAgent(ModelAgent):
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

        self.tgt_field = None
        self.src_field = "source"

        # define loss
        self.loss = self.contrastive_loss = ContrastiveLoss(
            metric=self.config.training.contrastive_loss.get("metric", "euclidean"),
            loss_type=self.config.training.contrastive_loss.get("loss_type", "softnn"),
            tau=self.config.training.contrastive_loss.get("tau", 1.0),
        )

        # define model

        self.model = RetrievalModel(self.config, self.input_tokenizer, src_field=self.src_field)

        # define optimizer
        if training_mode:
            self.create_optimizer()

        self.set_device(use_cuda)

        # self.create_samplers()

    def step_train(self, batch: Dict[str, torch.Tensor], tgt_field=None) -> torch.Tensor:
        batch["_global_step"] = self.global_step

        encodings, memory = self.model(batch)

        this_loss = torch.zeros(encodings.shape[0], dtype=torch.float).to(self.device)

        cont_loss = self.loss(encodings, batch[self.src_field + "_group"])

        Logger().log_scalar("train/contrastive_loss", cont_loss.mean(), self.global_step)

        this_loss += cont_loss

        if "loss" in memory:
            this_loss += memory["loss"]

        if self.config.training.get("loss_dropping", False):
            mask = self.dropper(this_loss)  # The dropper returns a mask of 0s where data should be dropped.
            this_loss *= mask  # Mask out the high losses')

        this_loss = torch.mean(this_loss, dim=0)

        return this_loss

    def step_validate(
        self,
        batch: Dict[str, torch.Tensor],
        tgt_field=None,
        sample_outputs=False,
        calculate_loss=True,
        reduce_outputs=True,
    ) -> Tuple[torch.Tensor, None, None, None, None, Dict[str, torch.Tensor]]:
        """
        Perform a single inference step
        """

        batch["_global_step"] = self.global_step

        encodings, memory = self.model(batch)

        if "vq_codes" in memory:
            for h_ix in range(memory["vq_codes"].shape[1]):
                self.vq_codes[h_ix].extend(memory["vq_codes"][:, h_ix].tolist())

        if calculate_loss:
            this_loss = torch.zeros(encodings.shape[0], dtype=torch.float).to(self.device)

            this_loss += self.loss(encodings, batch[self.src_field + "_group"])

            if "loss" in memory:
                this_loss += memory["loss"]

            this_loss = this_loss.mean(dim=0)
        else:
            this_loss = None

        return this_loss, None, None, None, None, memory
