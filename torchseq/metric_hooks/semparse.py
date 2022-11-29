import jsonlines
import torch
import numpy as np
import sacrebleu
import copy
import os
from abc import abstractmethod
from tqdm import tqdm

from collections import defaultdict
from torchseq.metric_hooks.base import MetricHook
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.metrics import bleu_corpus
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config

from torch.autograd import Variable
from torchseq.utils.functions import onehot, batchify


import logging

logger = logging.getLogger("SemParseMetricHook")


class SemanticParsingMetricHook(MetricHook):

    type = "slow"  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation

    # def __init__(self, config, src_field=None, tgt_field=None):
    #     super().__init__(config, src_field, tgt_field)

    def on_begin_epoch(self, use_test=False):
        self.scores = {}

    def on_batch(self, batch, logits, output, memory, use_test=False):
        pass

    def on_end_epoch(self, agent, use_test=False):
        # Temporarily change the config so the bottleneck is noiseless

        sample_outputs = self.config.data["eval"].get("sample_outputs", True)
        self.config.eval.data["sample_outputs"] = True

        if (
            self.config.eval.metrics.semparse.get("run_codepred", False)
            and self.config.bottleneck.get("code_predictor", None) is not None
        ):
            logger.info("Running generation with code prediction")
            self.scores["semparse_codepred_em"], codepred_output = SemanticParsingMetricHook.eval_gen_codepred(
                self.config,
                agent,
                test=use_test,
            )
            with open(
                os.path.join(agent.run_output_path, "codepred_output.{:}.txt".format("test" if use_test else "dev")),
                "w",
            ) as f:
                f.write("\n".join(codepred_output))
            logger.info("...done")

        # Reset the config
        self.config.eval.data["sample_outputs"] = sample_outputs
        return self.scores

    @abstractmethod
    def eval_gen_codepred(
        config,
        agent,
        test=False,
    ):
        sample_outputs = config.data["eval"].get("sample_outputs", True)

        # Now run eval

        infer_codes = agent.config.bottleneck.code_predictor.data.get("infer_codes", False)
        agent.config.bottleneck.code_predictor.data["infer_codes"] = True

        config_eval = copy.deepcopy(config.data)
        config_eval["dataset"] = "json"
        config_eval["json_dataset"] = {
            "path": "semparse/atis",
            "field_map": [
                {"type": "copy", "from": "target", "to": "target", "tokenizer": "output"},
                {"type": "copy", "from": "source", "to": "source"},
                {"type": "copy", "from": "target", "to": "template", "tokenizer": "output"},
            ],
        }
        config = Config(config_eval)

        data_loader = JsonDataLoader(data_path=agent.data_path, config=config)

        agent.config.eval.data["sample_outputs"] = True

        _, _, (output, _, _), _ = agent.inference(
            data_loader.test_loader if test else data_loader.valid_loader, desc="Generating"
        )

        agent.config.eval.data["sample_outputs"] = sample_outputs

        agent.config.bottleneck.code_predictor.data["infer_codes"] = infer_codes  # reset this flag

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(
                agent.data_path,
                config.json_dataset.path,
                f"{split}.jsonl",
            )
        ) as f:
            rows = [row for row in f][: config.eval.get("truncate_dataset", None)]

        refs = [row["target"] for row in rows]

        exact_match = np.mean([1 if pred == tgt else 0 for pred, tgt in zip(output, refs)])

        return exact_match, output
