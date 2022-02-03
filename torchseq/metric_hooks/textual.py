from collections import defaultdict
import torch
import numpy as np

from torchseq.metric_hooks.base import MetricHook
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.metrics import bleu_corpus, meteor_corpus, ibleu_corpus
from torchseq.utils.sari import SARIsent
import sacrebleu


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
        if len(output) > 0:
            if self.config.eval.data.get("topk", 1) > 1:
                self.pred_targets.extend([x[0] for x in output])
            else:
                self.pred_targets.extend(output)
            if "_refs_text" in batch:
                self.gold_targets.extend(batch["_refs_text"])
            else:
                self.gold_targets.extend([[x] for x in batch[self.tgt_field + "_text"]])
            self.inputs.extend(batch[self.src_field + "_text"])

    def on_end_epoch(self, _, use_test=False):

        # Flip and pad the references
        max_num_refs = max([len(x) for x in self.gold_targets])
        self.gold_targets = [x + [x[0]] * (max_num_refs - len(x)) for x in self.gold_targets]

        # print(len(self.gold_targets), len(self.pred_targets), len(self.inputs))

        self.scores["bleu"] = sacrebleu.corpus_bleu(
            self.pred_targets, list(zip(*self.gold_targets)), lowercase=True
        ).score
        self.scores["selfbleu"] = sacrebleu.corpus_bleu(self.pred_targets, [self.inputs], lowercase=True).score

        alpha = 0.8
        # self.scores["ibleu"] = ibleu_corpus(self.gold_targets, self.pred_targets, self.inputs)
        self.scores["ibleu"] = alpha * self.scores["bleu"] - (1 - alpha) * self.scores["selfbleu"]

        self.scores["sari"] = 100 * np.mean(
            [
                SARIsent(self.inputs[ix], self.pred_targets[ix], self.gold_targets[ix])
                for ix in range(len(self.pred_targets))
            ]
        )

        self.scores["em"] = np.mean(
            [self.pred_targets[ix] in self.gold_targets[ix] for ix in range(len(self.pred_targets))]
        )

        # self.scores["meteor"] = meteor_corpus(self.gold_targets, self.pred_targets)

        return self.scores
