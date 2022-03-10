import jsonlines
import torch
import numpy as np
import copy
import os
from abc import abstractmethod
from tqdm import tqdm

from collections import defaultdict, Counter
from torchseq.metric_hooks.base import MetricHook
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config
import sacrebleu


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

        if self.config.eval.metrics.hrq_agg.get("run_generate_summaries", False):
            logger.info("Running generation using HRQ paths")
            self.scores["hrq_agg"], _ = HRQAggregationMetricHook.eval_generate_summaries(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_masked_generation", False):
            logger.info("Running generation with masking")
            self.scores["hrq_agg"], _ = HRQAggregationMetricHook.eval_masked_generation(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        return self.scores

    @abstractmethod
    def eval_retrieval(config, agent, test=False):

        return {}, []

    @abstractmethod
    def eval_generate_summaries(config, agent, test=False):
        # First, get encodings for all sentences
        config_codes = copy.deepcopy(config.data)
        config_codes["dataset"] = "json"
        config_codes["json_dataset"] = {
            "path": config.eval.metrics.hrq_agg.dataset_all,
            "filename": "space_reviews.{split}",
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "s2"},
                {"type": "copy", "from": "sentence", "to": "s1"},
            ],
        }

        MASK_LENGTH = 3

        if agent.config.bottleneck.get("quantizer_heads", None) is not None:
            num_heads = agent.config.bottleneck.quantizer_heads - agent.config.bottleneck.get(
                "quantizer_num_residual", 0
            )
        else:
            bneck_types = [x.type for x in agent.config.bottleneck.modules]
            if "hrqvae" not in bneck_types:
                logger.warning("Tried to run hrq aggregation on a model without HRQ!")
                return {}, []
            quantizer_index = bneck_types.index("hrqvae")
            num_heads = agent.config.bottleneck.modules[quantizer_index].quantizer.num_heads

        data_loader = JsonDataLoader(config=Config(config_codes), data_path=agent.data_path)

        sample_outputs = agent.config.eval.get("sample_outputs", True)
        agent.config.eval.data["sample_outputs"] = False

        _, _, _, memory = agent.inference(
            data_loader.test_loader if test else data_loader.valid_loader, memory_keys_to_return=["vq_codes"]
        )

        all_codes = memory["vq_codes"]

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(agent.data_path, config.eval.metrics.hrq_agg.dataset_all, f"space_reviews.{split}.jsonl")
        ) as reader:
            all_rows = [x for x in reader]

        # Identify top clusters per entity
        codes_by_entity = defaultdict(Counter)

        for row, codes in zip(all_rows, all_codes):
            codes_by_entity[row["entity_id"]][tuple(codes.tolist()[:-MASK_LENGTH])] += 1

        mask = [1] * (num_heads - MASK_LENGTH) + [0] * MASK_LENGTH

        filtered_examples = []
        for entity, counter in codes_by_entity.items():
            for codes, count in counter.most_common(5):
                filtered_examples.append(
                    {"entity_id": entity, "codes": list(codes) + [0] * MASK_LENGTH, "sentence": "", "head_mask": mask}
                )

        # Generate!
        agent.config.eval.data["sample_outputs"] = True

        config_forced = copy.deepcopy(config.data)
        config_forced["dataset"] = "json"
        config_forced["json_dataset"] = {
            "path": config.eval.metrics.hrq_agg.dataset_all,
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "s2"},
                {"type": "copy", "from": "sentence", "to": "s1"},
                {"type": "copy", "from": "codes", "to": "forced_codes"},
                {"type": "copy", "from": "head_mask", "to": "head_mask"},
            ],
        }

        forced_loader = JsonDataLoader(config=Config(config_forced), dev_samples=filtered_examples)

        _, _, (output, _, _), _ = agent.inference(forced_loader.valid_loader)

        # TODO: eval against reference summaries

        agent.config.eval.data["sample_outputs"] = sample_outputs

        return {}, output

    @abstractmethod
    def eval_masked_generation(config, agent, test=False, dev_samples=None, test_samples=None, skip_scores=False):
        config_gen_masked = copy.deepcopy(config.data)
        config_gen_masked["dataset"] = "json"
        config_gen_masked["json_dataset"] = {
            "path": config.eval.metrics.hrq_agg.dataset_all,
            "filename": "space_reviews.{split}",
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "s2"},
                {"type": "copy", "from": "sentence", "to": "s1"},
                {"type": "copy", "from": "head_mask", "to": "head_mask"},
                {"type": "copy", "from": "residual_mask", "to": "residual_mask"},
            ],
        }

        data_loader = JsonDataLoader(
            data_path=agent.data_path,
            config=Config(config_gen_masked),
            dev_samples=dev_samples,
            test_samples=test_samples,
        )

        bneck_types = [x.type for x in agent.config.bottleneck.modules]
        if "hrqvae" not in bneck_types:
            logger.warning("Tried to run oracle masked eval on a model without a quantizer!")
            return {}
        quantizer_index = bneck_types.index("hrqvae")
        num_heads = agent.config.bottleneck.modules[quantizer_index].quantizer.num_heads

        scores = {}
        outputs = {}

        split = "test" if test else "dev"

        if not skip_scores:
            with jsonlines.open(
                os.path.join(
                    agent.data_path,
                    config.eval.metrics.hrq_agg.dataset_all,
                    f"space_reviews.{split}.jsonl",
                )
            ) as f:
                rows = [row for row in f][: config_gen_masked["eval"].get("truncate_dataset", None)]
            refs = [[q["sentence"]] for q in rows]

        for mask_length in range(0, num_heads + 1):
            mask = [1] * (num_heads - mask_length) + [0] * mask_length
            samples = (data_loader._test if test else data_loader._valid).samples
            samples = [{**x, "head_mask": mask, "residual_mask": [0]} for x in samples]
            masked_loader = JsonDataLoader(
                data_path=agent.data_path, config=Config(config_gen_masked), dev_samples=samples
            )

            _, _, (output, _, _), _ = agent.inference(masked_loader.valid_loader)

            if not skip_scores:
                # refs = [x["paras"] for x in qs_by_para_split]
                max_num_refs = max([len(x) for x in refs])
                refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

                tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded)), lowercase=True).score

                scores[mask_length] = tgt_bleu

            outputs[mask_length] = output

        return scores, outputs
