from typing import Dict, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn

from torchseq.agents.model_agent import ModelAgent

from torchseq.models.retrieval_model import RetrievalModel
from torchseq.models.contrastive_loss import ContrastiveLoss, NliContrastiveLoss
from torchseq.models.contrastive_triplet_loss import ContrastiveTripletLoss


from torchseq.utils.logging import Logger


class RetrievalAgent(ModelAgent):
    loss: Union[ContrastiveLoss, NliContrastiveLoss]
    src_field: str

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

        self.tgt_field = "query" if self.config.training.contrastive_loss.get("biencoder", False) else None
        self.src_field = "source"
        self.triplet_loss = False

        # define loss
        if self.config.training.contrastive_loss.get("nli_scores", False):
            self.loss = NliContrastiveLoss(
                metric=self.config.training.contrastive_loss.get("metric", "euclidean"),
                loss_type=self.config.training.contrastive_loss.get("loss_type", "softnn"),
                tau=self.config.training.contrastive_loss.get("tau", 1.0),
                src_field=self.src_field,
                bsz=self.config.training.contrastive_loss.get("nli_bsz", 64),
            )
        elif self.config.training.contrastive_loss.get("triplet_loss", False):
            self.loss = ContrastiveTripletLoss(
                metric=self.config.training.contrastive_loss.get("metric", "euclidean"),
                loss_type=self.config.training.contrastive_loss.get("loss_type", "softnn"),
                tau=self.config.training.contrastive_loss.get("tau", 1.0),
                inbatch_negatives=self.config.training.contrastive_loss.get("inbatch_negatives", False),
                softnn_agg_mean=self.config.training.contrastive_loss.get("softnn_agg_mean", False),
            )
            self.triplet_loss = True
            self.src_field = "query"
        else:
            self.loss = ContrastiveLoss(
                metric=self.config.training.contrastive_loss.get("metric", "euclidean"),
                loss_type=self.config.training.contrastive_loss.get("loss_type", "softnn"),
                tau=self.config.training.contrastive_loss.get("tau", 1.0),
            )

        # define model
        self.model = RetrievalModel(self.config, self.input_tokenizer)

        # define optimizer
        if training_mode:
            self.create_optimizer()

        self.set_device(use_cuda)

        # self.create_samplers()

    def step_train(self, batch: Dict[str, torch.Tensor], tgt_field=None) -> torch.Tensor:
        batch["_global_step"] = self.global_step

        encodings, memory = self.model(batch, src_field=self.src_field)

        encodings2, memory2, groups = None, None, None
        pos_encodings, pos_memory = None, None
        neg_encodings, neg_memory = None, None
        if self.config.training.contrastive_loss.get("biencoder", False):
            encodings2, memory2 = self.model(batch, src_field=self.tgt_field)
        elif self.config.training.contrastive_loss.get("triplet_loss", False):
            pos_encodings, pos_memory = self.model(batch, src_field="pos_target")
            neg_encodings, neg_memory = self.model(batch, src_field="neg_target")
        else:
            groups = batch[self.src_field + "_group"]

        this_loss = torch.zeros(encodings.shape[0], dtype=torch.float).to(self.device)

        if self.config.training.contrastive_loss.get("nli_scores", False):
            if encodings2 is not None:
                raise Exception("biencoder setup isnt supported for nli_scores yet!")
            cont_loss = self.loss(encodings, batch)
        elif self.config.training.contrastive_loss.get("triplet_loss", False):
            cont_loss = self.loss(
                encodings,
                pos_encodings=pos_encodings,
                neg_encodings=neg_encodings,
                pos_scores=batch.get("pos_score", None),
                neg_scores=batch.get("neg_score", None),
            )
        else:
            cont_loss = self.loss(encodings, encodings2, groups=groups)

        Logger().log_scalar("train/contrastive_loss", cont_loss.mean(), self.global_step)

        this_loss += cont_loss

        if "loss" in memory:
            this_loss += memory["loss"]

        if memory2 is not None and "loss" in memory2:
            this_loss += memory2["loss"]
        if pos_memory is not None and "loss" in pos_memory:
            this_loss += pos_memory["loss"]
        if neg_memory is not None and "loss" in neg_memory:
            this_loss += neg_memory["loss"]

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

        encodings, memory = self.model(batch, src_field=self.src_field)

        encodings2, memory2, groups = None, None, None
        pos_encodings, pos_memory = None, None
        neg_encodings, neg_memory = None, None
        if self.config.training.contrastive_loss.get("biencoder", False):
            encodings2, memory2 = self.model(batch, src_field=self.tgt_field)
        elif self.config.training.contrastive_loss.get("triplet_loss", False):
            if calculate_loss:
                pos_encodings, pos_memory = self.model(batch, src_field="pos_target")
                neg_encodings, neg_memory = self.model(batch, src_field="neg_target")
        else:
            groups = batch[self.src_field + "_group"]

        if "vq_codes" in memory:
            for h_ix in range(memory["vq_codes"].shape[1]):
                self.vq_codes[h_ix].extend(memory["vq_codes"][:, h_ix].tolist())

        if calculate_loss:
            this_loss = torch.zeros(encodings.shape[0], dtype=torch.float).to(self.device)

            if self.config.training.contrastive_loss.get("nli_scores", False):
                if encodings2 is not None:
                    raise Exception("biencoder setup isnt supported for nli_scores yet!")
                this_loss += self.loss(encodings, batch)
            elif self.config.training.contrastive_loss.get("triplet_loss", False):
                this_loss += self.loss(
                    encodings,
                    pos_encodings=pos_encodings,
                    neg_encodings=neg_encodings,
                    pos_scores=batch.get("pos_score", None),
                    neg_scores=batch.get("neg_score", None),
                )
            else:
                this_loss += self.loss(encodings, encodings2, groups=groups)

            if "loss" in memory:
                this_loss += memory["loss"]

            if memory2 is not None and "loss" in memory2:
                this_loss += memory2["loss"]

            this_loss = this_loss.mean(dim=0)
        else:
            this_loss = None

        return this_loss, None, None, None, None, memory
