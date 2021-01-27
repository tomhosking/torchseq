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
        self.scores["cluster_gen_with_templ_bleu"] = SepAEMetricHook.eval_gen_with_oracle(self.config, agent)
        self.logger.info("...done")

        self.logger.info("Running generation with noised encodings")
        (
            self.scores["cluster_gen_noised_diversity_bleu"],
            self.scores["cluster_gen_noised_diversity_selfbleu"],
            self.scores["cluster_gen_noised_diversity_ibleu"],
        ) = SepAEMetricHook.eval_gen_noised_diversity(self.config, agent)
        self.logger.info("...done")

        self.logger.info("Running generation to test reconstruction")
        self.scores["sepae_reconstruction"] = SepAEMetricHook.eval_reconstruction(self.config, agent)
        self.logger.info("...done")

        # Reset the config
        self.config.bottleneck.data["prior_var_weight"] = var_weight
        return self.scores

    @abstractmethod
    def eval_gen_with_oracle(config, agent, test=False, use_qqp=False):
        config_gen_with_templ = copy.deepcopy(config.data)
        config_gen_with_templ["dataset"] = "json"
        config_gen_with_templ["json_dataset"] = {
            "path": ("qqp-splitforgeneval" if use_qqp else "wikianswers-para-splitforgeneval"),
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "syn_input", "to": "template"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }
        config_gen_with_templ["eval"]["topk"] = 1

        config.bottleneck.data["prior_var_weight"] = 0.0

        data_loader = JsonDataLoader(config=Config(config_gen_with_templ))

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(
                config.env.data_path,
                (
                    f"qqp-splitforgeneval/{split}.jsonl"
                    if use_qqp
                    else f"wikianswers-para-splitforgeneval/{split}.jsonl"
                ),
            )
        ) as f:
            rows = [row for row in f][: config_gen_with_templ["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows]
        inputs = [[q["sem_input"]] for q in rows]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu = sacrebleu.corpus_bleu(output, list(zip(*inputs))).score

        alpha = 0.8
        ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

        return (tgt_bleu, self_bleu, ibleu)

    @abstractmethod
    def eval_reconstruction(config, agent, test=False):
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

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(config.env.data_path, f"wikianswers-para-splitforgeneval/{split}.jsonl")
        ) as f:
            refs = [[q["sem_input"]] for q in f][: config_reconstruction["eval"].get("truncate_dataset", None)]

        return sacrebleu.corpus_bleu(output, list(zip(*refs))).score

    @abstractmethod
    def eval_gen_noised_diversity(config, agent, noise_weight=2.0, code_offset=2, test=False):
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

        if config_gen_noised["bottleneck"]["vector_quantized"]:
            var_offset = config_gen_noised["bottleneck"]["quantizer_num_residual"]
            agent.model.bottleneck.quantizer._code_offset = (
                [0] * var_offset
                + [code_offset]
                + [code_offset] * (config_gen_noised["bottleneck"]["quantizer_heads"] - var_offset - 1)
            )
        else:
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

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(config.env.data_path, f"wikianswers-para-splitforgeneval/{split}.jsonl")
        ) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows for _ in range(5)]
        inputs = [[q["sem_input"]] for q in rows for _ in range(5)]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        config.bottleneck.data["prior_var_weight"] = 0.0
        agent.model.bottleneck.quantizer._code_offset = 0

        tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu = sacrebleu.corpus_bleu(output, list(zip(*inputs))).score

        alpha = 0.8
        ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

        return (tgt_bleu, self_bleu, ibleu)

    @abstractmethod
    def eval_gen_diversity_with_lookup(config, agent, test=False):
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

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        split = "test" if test else "dev"

        with jsonlines.open(os.path.join(config.env.data_path, f"wikianswers-para-exemplarlookup/{split}.jsonl")) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows]
        inputs = [[q["sem_input"]] for q in rows]
        # templates = [[q["syn_input"]] for q in rows]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        # config.bottleneck.data["prior_var_weight"] = 0.0

        tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu = sacrebleu.corpus_bleu(output, list(zip(*inputs))).score

        alpha = 0.8
        ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

        return (tgt_bleu, self_bleu, ibleu)

    @abstractmethod
    def eval_gen_diversity_with_cooccurence(config, agent, test=False):
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

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(config.env.data_path, f"wikianswers-para-exemplarcooccur/{split}.jsonl")
        ) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows]
        inputs = [[q["sem_input"]] for q in rows]
        # templates = [[q["syn_input"]] for q in rows]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        # config.bottleneck.data["prior_var_weight"] = 0.0

        tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu = sacrebleu.corpus_bleu(output, list(zip(*inputs))).score

        alpha = 0.8
        ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

        return (tgt_bleu, self_bleu, ibleu)

    @abstractmethod
    def eval_gen_diversity_with_nn(config, agent, test=False):
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": "wikianswers-para-exemplarnn",
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
                {"type": "copy", "from": "syn_input", "to": "template"},
            ],
        }
        config_gen_noised["eval"]["topk"] = 1

        data_loader = JsonDataLoader(config=Config(config_gen_noised))

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        split = "test" if test else "dev"

        with jsonlines.open(os.path.join(config.env.data_path, f"wikianswers-para-exemplarnn/{split}.jsonl")) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows]
        inputs = [[q["sem_input"]] for q in rows]
        # templates = [[q["syn_input"]] for q in rows]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        # config.bottleneck.data["prior_var_weight"] = 0.0

        tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu = sacrebleu.corpus_bleu(output, list(zip(*inputs))).score

        alpha = 0.8
        ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

        return (tgt_bleu, self_bleu, ibleu)

    @abstractmethod
    def eval_gen_diversity_with_code_lookup(config, agent, test=False):
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": "wikianswers-para-exemplarcodelookup",
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
                {"type": "copy", "from": "syn_input", "to": "template"},
                {"type": "copy", "from": "vq_codes", "to": "forced_codes"},
            ],
        }
        config_gen_noised["eval"]["topk"] = 1

        data_loader = JsonDataLoader(config=Config(config_gen_noised))

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(config.env.data_path, f"wikianswers-para-exemplarcodelookup/{split}.jsonl")
        ) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows]
        inputs = [[q["sem_input"]] for q in rows]
        # templates = [[q["syn_input"]] for q in rows]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        # config.bottleneck.data["prior_var_weight"] = 0.0

        tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu = sacrebleu.corpus_bleu(output, list(zip(*inputs))).score

        alpha = 0.8
        ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

        return (tgt_bleu, self_bleu, ibleu)

    @abstractmethod
    def eval_gen_diversity_with_mlp(config, agent, test=False, use_qqp=False):
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": ("qqp-exemplarmlppredict" if use_qqp else "wikianswers-para-exemplarmlppredict"),
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
                {"type": "copy", "from": "syn_input", "to": "template"},
                {"type": "copy", "from": "vq_codes", "to": "forced_codes"},
            ],
        }
        config_gen_noised["eval"]["topk"] = 1

        config.bottleneck.data["prior_var_weight"] = 0.0

        data_loader = JsonDataLoader(config=Config(config_gen_noised))

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        NUM_HYPOTHESES = 2

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(
                config.env.data_path,
                (
                    f"qqp-exemplarmlppredict/{split}.jsonl"
                    if use_qqp
                    else f"wikianswers-para-exemplarmlppredict/{split}.jsonl"
                ),
            )
        ) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]

        # First, do top-1
        refs_top1 = [q["paras"] for i, q in enumerate(rows) if i % NUM_HYPOTHESES == 0]
        inputs_top1 = [[q["sem_input"]] for i, q in enumerate(rows) if i % NUM_HYPOTHESES == 0]
        output_top1 = [q for i, q in enumerate(output) if i % NUM_HYPOTHESES == 0]

        max_num_refs = max([len(x) for x in refs_top1])
        refs_padded_top1 = [x + [x[0]] * (max_num_refs - len(x)) for x in refs_top1]

        # config.bottleneck.data["prior_var_weight"] = 0.0

        tgt_bleu_top1 = sacrebleu.corpus_bleu(output_top1, list(zip(*refs_padded_top1))).score
        self_bleu_top1 = sacrebleu.corpus_bleu(output_top1, list(zip(*inputs_top1))).score

        alpha = 0.8
        ibleu_top1 = alpha * tgt_bleu_top1 - (1 - alpha) * self_bleu_top1

        # Now, score the multiple outputs within each example
        refs = [q["paras"] for q in rows]
        N = NUM_HYPOTHESES
        # Get the other outputs from the same grouping
        other_outs = [
            [q for j, q in enumerate(output[N * (i // N) : N * (i // N + 1)]) if j % N != i % N]
            for i in range(len(output))
        ]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        tgt_bleu_div = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu_div = sacrebleu.corpus_bleu(output, list(zip(*other_outs))).score
        ibleu_div = alpha * tgt_bleu_div - (1 - alpha) * self_bleu_div

        return (tgt_bleu_top1, self_bleu_top1, ibleu_top1), (tgt_bleu_div, self_bleu_div, ibleu_div), output

    @abstractmethod
    def eval_gen_beam_diversity(config, agent, test=False):
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": "wikianswers-para-splitforgeneval",
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }

        prev_topk = config.eval.get("topk", 1)
        config.eval.data["topk"] = 4

        data_loader = JsonDataLoader(config=Config(config_gen_noised))

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        NUM_HYPOTHESES = 4

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(config.env.data_path, f"wikianswers-para-splitforgeneval/{split}.jsonl")
        ) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]

        output = [q for beam in output for q in beam]

        # Now, score the multiple outputs within each example
        refs = [q["paras"] for q in rows for _ in range(NUM_HYPOTHESES)]
        N = NUM_HYPOTHESES
        # Get the other outputs from the same grouping
        other_outs = [
            [q for j, q in enumerate(output[N * (i // N) : N * (i // N + 1)]) if j % N != i % N]
            for i in range(len(output))
        ]

        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        alpha = 0.8
        tgt_bleu_div = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu_div = sacrebleu.corpus_bleu(output, list(zip(*other_outs))).score
        ibleu_div = alpha * tgt_bleu_div - (1 - alpha) * self_bleu_div

        config.eval.data["topk"] = prev_topk

        return (tgt_bleu_div, self_bleu_div, ibleu_div)
