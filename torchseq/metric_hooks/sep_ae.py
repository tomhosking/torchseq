import jsonlines
import torch
import numpy as np
import sacrebleu
import copy
import os

from collections import defaultdict
from torchseq.metric_hooks.base import MetricHook
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.metrics import bleu_corpus
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config

import logging


class SepAEMetricHook(MetricHook):

    type = "slow"  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation

    def __init__(self, config, src_field=None, tgt_field=None):
        super().__init__(config, src_field, tgt_field)

        self.logger = logging.getLogger("SepAEMetric")

    def on_begin_epoch(self):
        self.scores = {
            "retrieval": None,
            "cluster_gen_with_templ_bleu": None,
            # "cluster_gen_no_templ_bleu": None,
            # "cluster_gen_no_templ_bleu": None,
        }

    def on_batch(self, batch, logits, output, memory):
        pass

    def on_end_epoch(self, agent):
        # Temporarily change the config so the bottleneck is noiseless
        var_weight = self.config.bottleneck.data.get("prior_var_weight", 1.0)
        self.config.bottleneck.data["prior_var_weight"] = 0.0

        self.logger.info("Running generation with template")
        self.eval_gen_with_templ(agent)
        self.logger.info("...done")

        # Reset the config
        self.config.bottleneck.data["prior_var_weight"] = var_weight
        return self.scores

    def eval_gen_with_templ(self, agent):
        config_gen_with_templ = copy.deepcopy(self.config.data)
        config_gen_with_templ["dataset"] = "json"
        config_gen_with_templ["json_dataset"] = {
            "path": "wikianswers-para-splitforgeneval",
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "syn_input", "to": "template"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }

        data_loader = JsonDataLoader(config=Config(config_gen_with_templ))

        _, _, output, _ = agent.validate(
            data_loader, save=False, force_save_output=False, save_model=False, slow_metrics=False
        )

        with jsonlines.open(
            os.path.join(self.config.env.data_path, "wikianswers-para-splitforgeneval/dev.jsonl")
        ) as f:
            qs_by_para_split = [q["paras"] for q in f]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in qs_by_para_split])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in qs_by_para_split]

        self.scores["cluster_gen_with_templ_bleu"] = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score

    def eval_gen_with_templ_diversity(self, agent):
        config_gen_with_templ = copy.deepcopy(self.config.data)
        config_gen_with_templ["dataset"] = "json"
        config_gen_with_templ["json_dataset"] = {
            "path": "wikianswers-para-splitforgeneval",
            "field_map": [
                {"type": "copy", "from": "q", "to": "s2"},
                {"type": "copy", "from": "q", "to": "s1"},
                {"type": "sample", "from": "paras", "to": "template"},
            ],
        }

        data_loader = JsonDataLoader(config=Config(config_gen_with_templ))

        _, _, output, _ = agent.validate(
            data_loader, save=False, force_save_output=False, save_model=False, slow_metrics=False
        )

        with jsonlines.open(
            os.path.join(self.config.env.data_path, "wikianswers-para-splitforgeneval/dev.jsonl")
        ) as f:
            qs_by_para_split = [q["paras"] for q in f]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in qs_by_para_split])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in qs_by_para_split]

        self.scores["cluster_gen_with_templ_bleu"] = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
