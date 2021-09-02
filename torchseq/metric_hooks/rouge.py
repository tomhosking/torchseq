from collections import defaultdict
from torchseq.metric_hooks.base import MetricHook
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.metrics import bleu_corpus, meteor_corpus, ibleu_corpus
from torchseq.utils.sari import SARIsent
import torch
import numpy as np

import rouge


class RougeMetricHook(MetricHook):

    type = "live"  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation

    def __init__(self, config, src_field=None, tgt_field=None):
        super().__init__(config, src_field, tgt_field)

    def on_begin_epoch(self, use_test=False):
        self.scores = {"r1": [], "r2": [], "rl": []}

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

    def on_end_epoch(self, _, use_test=False):

        evaluator = rouge.Rouge(
            metrics=[
                "rouge-l",
            ],
            max_n=4,
            limit_length=True,
            length_limit=100,
            length_limit_type="words",
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True,
        )

        scores = evaluator.get_scores(self.pred_targets, self.gold_targets)

        self.scores["rouge"] = scores

        return self.scores
