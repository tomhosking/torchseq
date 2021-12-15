import jsonlines
import torch
import numpy as np
import copy
import os
from abc import abstractmethod
from tqdm import tqdm

from collections import defaultdict
from torchseq.metric_hooks.base import MetricHook
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config


import logging

logger = logging.getLogger("HRQAggregationMetric")


class HRQAggregationMetricHook(MetricHook):

    type = "slow"  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation

    def __init__(self, config, src_field=None, tgt_field=None):
        super().__init__(config, src_field, tgt_field)

    def on_begin_epoch(self, use_test=False):
        self.scores = {}

    def on_batch(self, batch, logits, output, memory, use_test=False):
        pass

    def on_end_epoch(self, agent, use_test=False):
        # Temporarily change the config so the bottleneck is noiseless

        if self.config.eval.metrics.hrq_agg.get("run_retrieval", False):
            logger.info("Running retrieval using HRQ paths")
            self.scores["hrq_agg"], _ = HRQAggregationMetricHook.eval_retrieval(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        return self.scores

    @abstractmethod
    def eval_retrieval(config, agent, test=False):

        return {}, []
