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

from agents.model_agent import ModelAgent
from args import FLAGS as FLAGS
from datasets.cqa_triple import CQATriple
from datasets.loaders import get_embeddings, load_glove
from datasets.newsqa_loader import NewsqaDataLoader
from datasets.preprocessed_loader import PreprocessedDataLoader
from datasets.squad_dataset import SquadDataset
from datasets.squad_loader import SquadDataLoader
from models.aq_transformer import TransformerAqModel
from models.cross_entropy import CrossEntropyLossWithLS
from models.suppression_loss import SuppressionLoss
from utils.logging import add_to_log
from utils.mckenzie import update_mckenzie
from utils.metrics import bleu_corpus
from utils.misc import print_cuda_statistics
from utils.tokenizer import BPE


class AQAgent(ModelAgent):
    def __init__(self, config, run_id, silent=False):
        super().__init__(config, run_id, silent)

        self.src_field = "c"
        self.tgt_field = "q"

        # define loss
        if self.config.training.label_smoothing != "UNUSED" and self.config.training.label_smoothing > 1e-6:
            self.loss = CrossEntropyLossWithLS(
                ignore_index=BPE.pad_id, smooth_eps=self.config.training.label_smoothing, reduction="none"
            )
        else:
            self.loss = nn.CrossEntropyLoss(ignore_index=BPE.pad_id, reduction="none")

        # define models
        self.model = TransformerAqModel(config) #, loss=self.loss

        # define data_loader
        if self.config.training.use_preprocessed_data:
            self.data_loader = PreprocessedDataLoader(config=config)
        else:
            if self.config.training.dataset[:5] == "squad":
                self.data_loader = SquadDataLoader(config=config)
            elif self.config.training.dataset == "newsqa":
                self.data_loader = NewsqaDataLoader(config=config)
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

        output, logits, _ = self.decode_teacher_force(self.model, batch, "q")  # bsd

        this_loss = self.loss(logits.permute(0,2,1), batch['q'])

        if self.config.training.suppression_loss_weight > 0:
            this_loss += self.config.training.suppression_loss_weight * self.suppression_loss(logits, batch["a"])

        loss += torch.mean(torch.sum(this_loss, dim=1) / batch["q_len"].to(this_loss), dim=0)

        return loss

    def text_to_batch(self, x, device):

        x["q"] = ""

        return {
            k: (v.to(self.device) if k[-5:] != "_text" else v)
            for k, v in SquadDataset.pad_and_order_sequences(
                [
                    SquadDataset.to_tensor(
                        x,
                        sent_window=self.config.prepro.sent_window,
                        tok_window=self.config.prepro.tok_window,
                        o_tag=2 if self.config.prepro.bio_tagging else 1,
                        concat_ctxt_ans=self.config.prepro.concat_ctxt_ans,
                    )
                ]
            ).items()
        }

        # parsed_triple = CQATriple(x['c'], x['a'], x['a_pos'], "",
        #     sent_window=self.config.prepro.sent_window,
        #     tok_window=self.config.prepro.tok_window,
        #     o_tag=2 if self.config.prepro.bio_tagging else 1,
        #     concat_ctxt_ans=self.config.prepro.concat_ctxt_ans)

        # if self.config.prepro.concat_ctxt_ans:
        #     sample = {'c':  torch.LongTensor(parsed_triple.ctxt_as_ids() + [102] +  parsed_triple.ans_as_ids()).to(device),
        #             # 'q': torch.LongTensor(parsed_triple.q_as_ids()),
        #             'a': torch.LongTensor(parsed_triple.ans_as_ids()).to(device),
        #             'a_pos': torch.LongTensor([0 for i in range(len(parsed_triple._ctxt_doc))] + [0] + [1 for i in range(len(parsed_triple._ans_doc))]).to(device),
        #             'c_len': torch.LongTensor([len(parsed_triple._ctxt_doc) + len(parsed_triple._ans_doc) + 1]).to(device),
        #             'a_len': torch.LongTensor([len(parsed_triple._ans_doc)]).to(device),
        #             # 'q_len': torch.LongTensor([len(parsed_triple._q_doc)])
        #             }
        # else:
        #     sample = {'c': torch.LongTensor(parsed_triple.ctxt_as_ids()).to(device),
        #             # 'q': torch.LongTensor(parsed_triple.q_as_ids()),
        #             'a': torch.LongTensor(parsed_triple.ans_as_ids()).to(device),
        #             'a_pos': torch.LongTensor(parsed_triple.ctxt_as_bio()).to(device),
        #             'c_len': torch.LongTensor([len(parsed_triple._ctxt_doc)]).to(device),
        #             'a_len': torch.LongTensor([len(parsed_triple._ans_doc)]).to(device),
        #             # 'q_len': torch.LongTensor([len(parsed_triple._q_doc)])
        #             }

        # return self.pad_and_order_sequences([sample])
