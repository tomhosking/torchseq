import jsonlines
import torch
import numpy as np
import sacrebleu
import copy
import os
from abc import abstractmethod

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

        self.logger.info("Running generation with oracle template")
        self.scores["cluster_gen_with_templ_bleu"] = SepAEMetricHook.eval_gen_with_templ(self.config, agent)
        self.logger.info("...done")

        self.logger.info("Running generation with noised encodings")
        (
            self.scores["cluster_gen_noised_diversity_bleu"],
            self.scores["cluster_gen_noised_diversity_selfbleu"],
        ) = SepAEMetricHook.eval_gen_noised_diversity(self.config, agent)
        self.logger.info("...done")

        self.logger.info("Running generation with tgt as exemplar")
        self.scores["sepae_reconstruction"] = SepAEMetricHook.eval_reconstruction(self.config, agent)
        self.logger.info("...done")

        # Reset the config
        self.config.bottleneck.data["prior_var_weight"] = var_weight
        return self.scores

    @abstractmethod
    def eval_gen_with_templ(config, agent):
        config_gen_with_templ = copy.deepcopy(config.data)
        config_gen_with_templ["dataset"] = "json"
        config_gen_with_templ["json_dataset"] = {
            "path": "wikianswers-para-splitforgeneval",
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "syn_input", "to": "template"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }
        config_gen_with_templ["eval"]["topk"] = 1

        data_loader = JsonDataLoader(config=Config(config_gen_with_templ))

        _, _, output, _ = agent.validate(
            data_loader, save=False, force_save_output=False, save_model=False, slow_metrics=False
        )

        with jsonlines.open(os.path.join(config.env.data_path, "wikianswers-para-splitforgeneval/dev.jsonl")) as f:
            refs = [q["paras"] for q in f][: config_gen_with_templ["eval"].get("truncate_dataset", None)]

        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]
        refs_transpose = list(zip(*refs_padded))

        # print(max_num_refs)
        # print(config_gen_with_templ['eval']['truncate_dataset'])
        # print(len(refs_transpose))
        # print(len(refs_transpose[0]))
        # print(refs_transpose)

        return sacrebleu.corpus_bleu(output, refs_transpose).score

    @abstractmethod
    def eval_reconstruction(config, agent):
        config_reconstruction = copy.deepcopy(config.data)
        config_reconstruction["dataset"] = "json"
        config_reconstruction["json_dataset"] = {
            "path": "wikianswers-para-splitforgeneval",
            "field_map": [
                {"type": "copy", "from": "sem_input", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "template"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }
        config_reconstruction["eval"]["topk"] = 1

        data_loader = JsonDataLoader(config=Config(config_reconstruction))

        _, _, output, _ = agent.validate(
            data_loader, save=False, force_save_output=False, save_model=False, slow_metrics=False
        )

        with jsonlines.open(os.path.join(config.env.data_path, "wikianswers-para-splitforgeneval/dev.jsonl")) as f:
            refs = [[q["sem_input"]] for q in f][: config_reconstruction["eval"].get("truncate_dataset", None)]

        return sacrebleu.corpus_bleu(output, list(zip(*refs))).score

    @abstractmethod
    def eval_gen_noised_diversity(config, agent, noise_weight=2.0):
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": "wikianswers-para-splitforgeneval",
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }
        config_gen_noised["eval"]["topk"] = 1
        config_gen_noised["eval"]["repeat_samples"] = 5

        var_offset = config_gen_noised["bottleneck"]["num_similar_heads"]
        if config_gen_noised["bottleneck"].get("invert_templ", False):
            var1 = noise_weight
            var2 = 0.0
        else:
            var1 = 0.0
            var2 = noise_weight
        config.bottleneck.data["prior_var_weight"] = (
            [var1] * var_offset + [var2] + [var2] * (config_gen_noised["encdec"]["num_heads"] - var_offset - 1)
        )

        data_loader = JsonDataLoader(config=Config(config_gen_noised))

        _, _, output, _ = agent.validate(
            data_loader, save=False, force_save_output=False, save_model=False, slow_metrics=False
        )

        with jsonlines.open(os.path.join(config.env.data_path, "wikianswers-para-splitforgeneval/dev.jsonl")) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows for _ in range(5)]
        inputs = [[q["sem_input"]] for q in rows for _ in range(5)]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        config.bottleneck.data["prior_var_weight"] = 0.0

        return (
            sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score,
            sacrebleu.corpus_bleu(output, list(zip(*inputs))).score,
        )

    @abstractmethod
    def eval_gen_diversity_with_lookup(config, agent):
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": "wikianswers-para-exemplarlookup",
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
                {"type": "copy", "from": "syn_input", "to": "template"},
            ],
        }
        config_gen_noised["eval"]["topk"] = 1

        data_loader = JsonDataLoader(config=Config(config_gen_noised))

        _, _, output, _ = agent.validate(
            data_loader, save=False, force_save_output=False, save_model=False, slow_metrics=False
        )

        with jsonlines.open(os.path.join(config.env.data_path, "wikianswers-para-exemplarlookup/dev.jsonl")) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows]
        inputs = [[q["sem_input"]] for q in rows]
        # templates = [[q["syn_input"]] for q in rows]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        # config.bottleneck.data["prior_var_weight"] = 0.0

        return (
            sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score,
            sacrebleu.corpus_bleu(output, list(zip(*inputs))).score,
        )

    @abstractmethod
    def eval_gen_diversity_with_cooccurence(config, agent):
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": "wikianswers-para-exemplarcooccur",
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
                {"type": "copy", "from": "syn_input", "to": "template"},
            ],
        }
        config_gen_noised["eval"]["topk"] = 1

        data_loader = JsonDataLoader(config=Config(config_gen_noised))

        _, _, output, _ = agent.validate(
            data_loader, save=False, force_save_output=False, save_model=False, slow_metrics=False
        )

        with jsonlines.open(os.path.join(config.env.data_path, "wikianswers-para-exemplarcooccur/dev.jsonl")) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows]
        inputs = [[q["sem_input"]] for q in rows]
        # templates = [[q["syn_input"]] for q in rows]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        # config.bottleneck.data["prior_var_weight"] = 0.0

        return (
            sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score,
            sacrebleu.corpus_bleu(output, list(zip(*inputs))).score,
        )
