from typing import Any, Optional

import jsonlines, os, json

import numpy as np
from collections import defaultdict

from torchseq.eval.recipes import EvalRecipe
from torchseq.utils.rouge import get_jackknife_rouge


class Recipe(EvalRecipe):
    name: str = "opagg.twostage_post"

    def run(self, predicted_summaries: Optional[list[str]] = None) -> dict[str, Any]:
        result = {}

        if predicted_summaries is None:
            # Load the input clusters
            with open(os.path.join(self.model_path, "eval", f"summaries_{self.split_str}.json")) as f:
                extractive_summaries = json.load(f)

            # Load the LLM outputs
            with jsonlines.open(
                os.path.join(self.model_path, "eval", f"llm_outputs_{self.split_str}.jsonl")
            ) as reader:
                llm_outputs = list(reader)

            # Clean and combine into summaries
            # TODO: label each output with its origin so that we don't have to do this
            i = 0
            predicted_summaries = []
            for clusters in extractive_summaries["evidence"]:
                curr_summ = []
                for cluster in clusters:
                    curr_summ.append(llm_outputs[i]["response"].strip())
                    i += 1
                predicted_summaries.append(" ".join(curr_summ))
        else:
            # Allow this recipe to be used for external systems (ie baselines)
            print("Using external summaries passed to eval recipe")

        # Load the references
        dataset_eval = self.config.eval.metrics.self_retrieval.get("dataset_eval", "opagg/space-eval")
        with jsonlines.open(os.path.join(self.data_path, dataset_eval, f"{self.split_str}.jsonl")) as reader:
            eval_data = [x for x in reader]

        if "space" in dataset_eval:
            with open("/mnt/ext/phd/data/space/space_summ.json") as f:
                space = json.load(f)
            product_names = {row["entity_id"]: row["entity_name"] for row in space}
            trivial_template = "I stayed at {:}."
        else:
            # TODO: can we get the product names for amasum?
            product_names = defaultdict(lambda: "this product")
            trivial_template = "I bought {:}."

        # Score the summaries

        # Rouge
        result["rouge"] = get_jackknife_rouge(predicted_summaries, [row["summaries"] for row in eval_data])

        # SummaC
        from summac.model_summac import SummaCConv

        # model_conv = SummaCConv(
        #     models=["vitc"],
        #     bins="percentile",
        #     granularity="sentence",
        #     nli_labels="e",
        #     device="cuda",
        #     start_file="default",
        #     agg="mean",
        # )

        # print("Evaling SC_ins")
        # docs = [" ".join([" ".join(sent for sent in rev["sentences"]) for rev in ent["reviews"]]) for ent in eval_data]
        # res = model_conv.score(docs, predicted_summaries)
        # result["sc_ins"] = np.mean(res["scores"]) * 100

        # print("Evaling SC_refs")
        # docs = [" ".join(row["summaries"]) for row in eval_data]
        # res = model_conv.score(docs, predicted_summaries)
        # result["sc_refs"] = np.mean(res["scores"]) * 100

        # # Prevalence
        print("Evaling prevalence")
        from torchseq.metric_hooks.prevalence_metric import PrevalenceMetric

        prevmet = PrevalenceMetric()
        prevs, reds, trivs = prevmet.get_prevalence(
            [[" ".join(rev["sentences"]) for rev in row["reviews"]] for row in eval_data],
            predicted_summaries,
            pbar=False,
            product_names=[product_names[row["entity_id"]] for row in eval_data],
            trivial_template=trivial_template,
        )
        result["prevalence"] = (np.mean(prevs), np.mean(reds), np.mean(trivs))

        return result
