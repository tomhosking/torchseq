from collections import defaultdict
from torchseq.metric_hooks.base import MetricHook
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.metrics import bleu_corpus, meteor_corpus, ibleu_corpus
from torchseq.utils.sari import SARIsent
import torch
import numpy as np


class TextualMetricHook(MetricHook):

    type = "live"  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation

    def __init__(self, config, src_field=None, tgt_field=None):
        super().__init__(config, src_field, tgt_field)

    def on_begin_epoch(self, use_test=False):
        self.scores = {"bleu": [], "meteor": [], "em": [], "sari": [], "ibleu": []}

        self.gold_targets = []
        self.pred_targets = []
        self.inputs = []

    def on_batch(self, batch, logits, output, memory, use_test=False):

        if self.config.eval.data.get("topk", 1) > 1:
            self.pred_targets.extend([x[0] for x in output])
        else:
            self.pred_targets.extend(output)
        self.gold_targets.extend(batch[self.tgt_field + "_text"])
        self.inputs.extend(batch[self.src_field + "_text"])

        # print(len(self.pred_targets))
        # print(len(self.gold_targets))
        # print(len(self.inputs))
        # exit()

    def on_end_epoch(self, _, use_test=False):

        # print(len(self.gold_targets), len(self.pred_targets), len(self.inputs))

        self.scores["bleu"] = bleu_corpus(self.gold_targets, self.pred_targets)
        self.scores["selfbleu"] = bleu_corpus(self.inputs, self.pred_targets)

        self.scores["ibleu"] = ibleu_corpus(self.gold_targets, self.pred_targets, self.inputs)

        self.scores["sari"] = 100 * np.mean(
            [
                SARIsent(self.inputs[ix], self.pred_targets[ix], [self.gold_targets[ix]])
                for ix in range(len(self.pred_targets))
            ]
        )

        self.scores["em"] = np.mean(
            [self.gold_targets[ix] == self.pred_targets[ix] for ix in range(len(self.pred_targets))]
        )

        self.scores["meteor"] = meteor_corpus(self.gold_targets, self.pred_targets)

        return self.scores
