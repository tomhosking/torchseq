from typing import Any, Optional, Literal

import jsonlines, os, json

import numpy as np
from collections import defaultdict

from torchseq.eval.recipes import EvalRecipe
from torchseq.utils.rouge import get_jackknife_rouge


class Recipe(EvalRecipe):
    name: str = "opagg.twostage_post"

    def run(
        self,
        predicted_summaries: Optional[list[str]] = None,
        prev_model: Literal["vitc", "vitc-base", "mnli", "mnli-base"] = "vitc",
        variant: Literal["oneshot", "sentencewise", "extractive"] = "extractive",
        silent: bool = False,
    ) -> dict[str, Any]:
        result = {}

        if not silent:
            print("Variant: ", variant)

        if predicted_summaries is None:
            # Load the input clusters
            with open(os.path.join(self.model_path, "eval", f"summaries_{self.split_str}.json")) as f:
                extractive_summaries = json.load(f)

            # Load the LLM outputs
            with jsonlines.open(
                os.path.join(self.model_path, "eval", f"llm_outputs_{variant}_{self.split_str}.jsonl")
            ) as reader:
                llm_outputs = list(reader)

            # Clean and combine into summaries
            # TODO: label each output with its origin so that we don't have to do this
            if variant == "sentencewise":
                i = 0
                predicted_summaries = []
                for clusters in extractive_summaries["evidence"]:
                    curr_summ = []
                    for cluster in clusters:
                        curr_summ.append(llm_outputs[i]["response"].strip())
                        i += 1
                    predicted_summaries.append(" ".join(curr_summ))
            elif variant == "oneshot":
                predicted_summaries = [resp["response"] for resp in llm_outputs]
            elif variant == "extractive":
                predicted_summaries = extractive_summaries["extractive_summaries"]
        else:
            # Allow this recipe to be used for external systems (ie baselines)
            if not silent:
                print("Using external summaries passed to eval recipe")

        # Load the references
        dataset_eval = self.config.eval.metrics.opsumm_cluster_aug.get("dataset_eval", "opagg/space-eval")
        with jsonlines.open(os.path.join(self.data_path, dataset_eval, f"{self.split_str}.jsonl")) as reader:
            eval_data = [x for x in reader]

        if "space" in dataset_eval:
            trivial_template = "I stayed at {:}."
            # trivial_template = "I stayed at this hotel."
        else:
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
        if not silent:
            print("Evaling prevalence")
        from torchseq.metric_hooks.prevalence_metric import PrevalenceMetric

        review_limit = 500

        prevmet = PrevalenceMetric(model_name=prev_model)
        adjusted_prevalence, (prevs, reds, trivs, gens), _ = prevmet.get_prevalence(
            [[" ".join(rev["sentences"]) for rev in row["reviews"][-review_limit:]] for row in eval_data],
            predicted_summaries,
            pbar=False,
            product_names=[row["entity_name"] for row in eval_data],
            trivial_template=trivial_template,
            include_generics=True,
        )
        result["prevalence"] = adjusted_prevalence * 100, (
            np.mean(prevs) * 100,
            np.mean(reds) * 100,
            np.mean(trivs) * 100,
            np.mean(gens) * 100,
        )

        return result
