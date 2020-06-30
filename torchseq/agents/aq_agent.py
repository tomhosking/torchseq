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
from torchseq.models.pretrained_modular import PretrainedModularModel
from torchseq.models.cross_entropy import CrossEntropyLossWithLS
from torchseq.models.suppression_loss import SuppressionLoss

from torchseq.utils.mckenzie import update_mckenzie
from torchseq.utils.tokenizer import Tokenizer


class AQAgent(ModelAgent):
    def __init__(self, config, run_id, output_path, silent=False, training_mode=True):
        super().__init__(config, run_id, output_path, silent, training_mode)

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
        if self.config.data.get("model", None) is not None and self.config.model == "pretrained_modular":
            self.model = PretrainedModularModel(self.config, src_field=self.src_field)
        else:
            self.model = TransformerAqModel(self.config)  # , loss=self.loss

        # define data_loader
        if self.config.training.use_preprocessed_data:
            self.data_loader = PreprocessedDataLoader(config=config)
        else:
            if (
                self.config.training.dataset
                in ["squad", "newsqa", "msmarco", "naturalquestions", "drop", "nq_newsqa", "squad_nq_newsqa"]
                or self.config.training.dataset[:5] == "squad"
            ):
                self.data_loader = QADataLoader(config=config)
            # elif self.config.training.dataset == 'newsqa':
            #     self.data_loader = NewsqaDataLoader(config=config)
            else:
                raise Exception("Unrecognised dataset: {:}".format(config.training.dataset))

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
        loss = 0

        output, logits, _, _ = self.decode_teacher_force(self.model, batch, "q")  # bsd

        this_loss = self.loss(logits.permute(0, 2, 1), batch["q"])

        if self.config.training.suppression_loss_weight > 0:
            this_loss += self.config.training.suppression_loss_weight * self.suppression_loss(logits, batch["a"])

        loss += torch.mean(torch.sum(this_loss, dim=1) / (batch["q_len"] - 1).to(this_loss), dim=0)

        return loss

    def text_to_batch(self, x, device):

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
