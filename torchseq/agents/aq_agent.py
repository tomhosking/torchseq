import json
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

from torchseq.agents.model_agent import ModelAgent

from torchseq.datasets.qa_triple import QATriple


from torchseq.datasets.preprocessed_loader import PreprocessedDataLoader
from torchseq.datasets.qa_dataset import QADataset
from torchseq.datasets.qa_loader import QADataLoader
from torchseq.models.aq_transformer import TransformerAqModel
from torchseq.models.pretrained_adapter import PretrainedAdapterModel
from torchseq.models.cross_entropy import CrossEntropyLossWithLS
from torchseq.models.suppression_loss import SuppressionLoss

from torchseq.utils.mckenzie import update_mckenzie
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.loss_dropper import LossDropper


class AQAgent(ModelAgent):
    def __init__(self, config, run_id, output_path, silent=False, training_mode=True, verbose=True):
        super().__init__(config, run_id, output_path, silent, training_mode, verbose)

        self.src_field = "c"
        self.tgt_field = "q"

        # define loss
        if self.config.training.label_smoothing != "UNUSED" and self.config.training.label_smoothing > 1e-6:
            self.loss = CrossEntropyLossWithLS(
                ignore_index=Tokenizer().pad_id, smooth_eps=self.config.training.label_smoothing, reduction="none"
            )
        else:
            self.loss = nn.CrossEntropyLoss(ignore_index=Tokenizer().pad_id, reduction="none")

        # define models
        if self.config.data.get("model", None) is not None and self.config.model == "pretrained_adapter":
            self.model = PretrainedAdapterModel(self.config, src_field=self.src_field, tgt_field=self.tgt_field)
        else:
            self.model = TransformerAqModel(self.config)  # , loss=self.loss

        self.suppression_loss = SuppressionLoss(self.config)

        # define optimizer
        self.create_optimizer()

        self.set_device()

        if self.cuda:
            self.suppression_loss = self.suppression_loss.to(self.device)

        self.create_samplers()

        self.begin_epoch_hook()

    def begin_epoch_hook(self):
        self.model.freeze_bert = self.current_epoch < self.config.encdec.bert_warmup_epochs

    def step_train(self, batch, tgt_field):
        batch["_global_step"] = self.global_step

        output, logits, _, memory = self.decode_teacher_force(self.model, batch, "q")  # bsd

        this_loss = torch.zeros(output.shape[0], dtype=torch.float).to(self.device)

        if self.config.training.get("xe_loss", True):
            xe_loss = self.loss(logits.permute(0, 2, 1), batch[tgt_field])
            # This isn't necessary - the teacher force decoder already forces the first token
            # if self.config.training.get("loss_offset", 0) > 0:
            #     xe_offset = self.config.training.get("loss_offset", 0)
            #     # xe_mask = torch.ones_like(xe_loss)
            #     # xe_mask[:, :xe_offset] = 0
            #     xe_loss = xe_loss[:, xe_offset:]
            #     # print(torch.argmax(logits, dim=-1)[0])
            #     # print(batch[tgt_field][0].shape)
            #     # print(xe_loss[0].shape)
            #     # exit()
            this_loss += xe_loss.sum(dim=1) / (batch[tgt_field + "_len"] - 1).to(this_loss)

        if self.config.training.suppression_loss_weight > 0:
            this_loss += (
                self.config.training.suppression_loss_weight
                * self.suppression_loss(logits, batch["a"]).sum(dim=1)
                / (batch["q_len"] - 1).to(this_loss)
            )

        if "loss" in memory:
            this_loss += memory["loss"]

        if self.config.training.get("loss_dropping", False):
            mask = self.dropper(this_loss)  # The dropper returns a mask of 0s where data should be dropped.
            this_loss *= mask  # Mask out the high losses')

        this_loss = torch.mean(this_loss, dim=0)

        return this_loss

    def text_to_batch(self, x, device):

        if "q" not in x:
            x["q"] = ""

        return {
            k: (v.to(self.device) if k[-5:] != "_text" else v)
            for k, v in QADataset.pad_and_order_sequences(
                [
                    QADataset.to_tensor(
                        x,
                        sent_window=self.config.prepro.sent_window,
                        tok_window=self.config.prepro.tok_window,
                        o_tag=2 if self.config.prepro.bio_tagging else 1,
                        concat_ctxt_ans=self.config.prepro.concat_ctxt_ans,
                    )
                ]
            ).items()
        }
