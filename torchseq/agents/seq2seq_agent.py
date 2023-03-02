import torch
import torch.nn.functional as F
from torch import nn

from torchseq.agents.model_agent import ModelAgent

from torchseq.models.bottleneck_autoencoder import BottleneckAutoencoderModel
from torchseq.models.pretrained_adapter import PretrainedAdapterModel
from torchseq.models.suppression_loss import SuppressionLoss
from torchseq.utils.loss_dropper import LossDropper


class Seq2SeqAgent(ModelAgent):
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

        if self.config.get("bottleneck", {}).get("use_templ_encoding", False):
            self.logger.warning("Using a Seq2Seq agent for exemplar guided tasks is deprecated!")

        self.tgt_field = "target" if not self.config.training.data.get("flip_pairs", False) else "source"

        if self.config.training.dataset is None:
            self.src_field = "source"
        elif self.config.training.dataset in [
            "squad",
            "squad_nq_newsqa",
            "nq_newsqa",
            "newsqa",
            "naturalquestions",
        ]:

            self.src_field = "q"
            self.tgt_field = "q"
        else:
            self.src_field = (
                "target"
                if (self.config.task == "autoencoder" or self.config.training.data.get("flip_pairs", False))
                else "source"
            )

        # define loss
        self.loss = nn.CrossEntropyLoss(
            ignore_index=self.output_tokenizer.pad_id,
            reduction="none",
            label_smoothing=self.config.training.get("label_smoothing", 0.0),
        )

        # define model
        if self.config.data.get("model", None) is not None and self.config.model == "pretrained_adapter":
            self.logger.warning(
                "PretrainedAdapterModel is deprecated! The standard model now supports pretrained encoders and decoders"
            )
            self.model = PretrainedAdapterModel(
                self.config,
                self.input_tokenizer,
                self.output_tokenizer,
                src_field=self.src_field,
                tgt_field=self.tgt_field,
            )
        else:
            self.model = BottleneckAutoencoderModel(
                self.config, self.input_tokenizer, self.output_tokenizer, src_field=self.src_field
            )

        self.suppression_loss = SuppressionLoss(self.config)

        if self.config.training.get("loss_dropping", 0) > 0:
            self.dropper = LossDropper(dropc=self.config.training.get("loss_dropping", 0), recompute=5000)

        # define optimizer
        if training_mode:
            self.create_optimizer()

        self.set_device(use_cuda)

        self.create_samplers()

    def step_train(self, batch, tgt_field):
        batch["_global_step"] = self.global_step

        output, logits, _, memory = self.decode_teacher_force(self.model, batch, tgt_field)

        this_loss = torch.zeros(output.shape[0], dtype=torch.float).to(self.device)

        if self.config.training.get("xe_loss", True):
            elementwise_loss = self.loss(logits.permute(0, 2, 1), batch[tgt_field])
            if elementwise_loss.isnan().any():
                print(elementwise_loss)
                print(logits[0, 1, :])
                raise Exception("NaN loss detected! Aborting training")

            this_loss += elementwise_loss[:, 1:].sum(dim=1) / (batch[tgt_field + "_len"] - 1).to(this_loss)

        if self.config.training.suppression_loss_weight > 0:
            this_loss += (
                self.config.training.suppression_loss_weight
                * self.suppression_loss(logits, batch["a"], self.output_tokenizer.pad_id).sum(dim=1)
                / (batch["q_len"] - 1).to(this_loss)
            )

        if "loss" in memory:
            this_loss += memory["loss"]

        if self.config.training.get("loss_dropping", False):
            mask = self.dropper(this_loss)  # The dropper returns a mask of 0s where data should be dropped.
            this_loss *= mask  # Mask out the high losses')

        this_loss = torch.mean(this_loss, dim=0)

        return this_loss
