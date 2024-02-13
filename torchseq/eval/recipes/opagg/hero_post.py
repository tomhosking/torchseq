from typing import Any, Optional, Literal

import jsonlines, os, json

import numpy as np
from collections import defaultdict

from torchseq.eval.recipes import EvalRecipe
from torchseq.utils.rouge import get_jackknife_rouge

from nltk.tokenize import sent_tokenize, word_tokenize


class Recipe(EvalRecipe):
    name: str = "opagg.twostage_post"

    def run(
        self,
        predicted_summaries: Optional[list[str]] = None,
        prev_model: Literal["vitc", "vitc-base", "mnli", "mnli-base"] = "vitc",
        variant: Literal["oneshot", "piecewise", "extractive"] = "extractive",
        llm_name: str = "llama7b",
        silent: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if not silent:
            print("Variant: ", variant)

        if predicted_summaries is None:
            # Load the input clusters
            with open(os.path.join(self.model_path, "eval", f"summaries_{self.split_str}.json")) as f:
                extractive_summaries = json.load(f)

            # Clean and combine into summaries
            # TODO: label each output with its origin so that we don't have to do this
            if variant == "extractive":
                predicted_summaries = extractive_summaries["extractive_summaries"]
                output_name = f"extractive_{self.split_str}"
            else:
                # Load the LLM outputs
                with jsonlines.open(
                    os.path.join(self.model_path, "eval", f"llm_outputs_{variant}_{self.split_str}_{llm_name}.jsonl")
                ) as reader:
                    llm_outputs = list(reader)

                output_name = f"{variant}_{self.split_str}_{llm_name}"

                if variant == "oneshot":
                    predicted_summaries = [self.cleanup_llm_output(resp["response"]) for resp in llm_outputs]

                elif variant == "piecewise":
                    i = 0
                    predicted_summaries = []
                    for clusters in extractive_summaries["evidence"]:
                        curr_summ = []
                        for cluster in clusters:
                            sent = self.cleanup_llm_output(llm_outputs[i]["response"])
                            curr_summ.append(sent)
                            i += 1
                        predicted_summaries.append(" ".join(curr_summ))

            with open(os.path.join(self.model_path, "eval", f"hero_{output_name}.txt"), "w") as f:
                f.writelines([summ + "\n" for summ in predicted_summaries])
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

        result["word_count"] = np.mean([len(word_tokenize(summ)) for summ in predicted_summaries])
        result["sent_count"] = np.mean([len(sent_tokenize(summ)) for summ in predicted_summaries])

        # Rouge
        result["rouge"] = get_jackknife_rouge(predicted_summaries, [row["summaries"] for row in eval_data])

        # SummaC
        # from summac.model_summac import SummaCConv

        # model_conv = SummaCConv(
        #     models=["vitc"],
        #     bins="percentile",
        #     granularity="sentence",
        #     nli_labels="e",
        #     device="cuda",
        #     start_file="default",
        #     agg="mean",
        # )

        # for imager in model_conv.imagers:
        #     imager.cache_folder = os.path.expanduser("~/.summac_cache/")
        #     os.makedirs(imager.cache_folder, exist_ok=True)
        #     imager.load_cache()

        # print("Evaling SC_ins")
        # docs = [" ".join([" ".join(sent for sent in rev["sentences"]) for rev in ent["reviews"]]) for ent in eval_data]
        # res = model_conv.score(docs, predicted_summaries)
        # result["sc_ins"] = np.mean(res["scores"]) * 100

        # print("Evaling SC_refs")
        # docs = [" ".join(row["summaries"]) for row in eval_data]
        # res = model_conv.score(docs, predicted_summaries)
        # result["sc_refs"] = np.mean(res["scores"]) * 100

        # if variant == 'piecewise':
        #     if not silent:
        #         print("Evaling attribution")
        #     docs = [" ".join(cluster) for clusters in extractive_summaries["evidence"] for cluster in clusters]
        #     preds = [pred['response'] for pred in llm_outputs]
        #     assert len(docs) == len(preds)
        #     res = model_conv.score(docs, preds)
        #     result["sc_attr"] = np.mean(res["scores"]) * 100
        # elif variant == 'oneshot':
        #     if not silent:
        #         print("Evaling attribution")
        #     docs = [" ".join([sent for cluster in clusters for sent in cluster]) for clusters in extractive_summaries["evidence"]]
        #     preds = predicted_summaries
        #     assert len(docs) == len(preds)
        #     res = model_conv.score(docs, preds)
        #     result["sc_attr"] = np.mean(res["scores"]) * 100
        # else:
        #     result['sc_attr'] = None

        # for imager in model_conv.imagers:
        #     imager.save_cache()

        # # Prevalence
        if not silent:
            print("Evaling prevalence")
        from torchseq.metric_hooks.prevalence_metric import PrevalenceMetric

        review_limit = 200

        prevmet = PrevalenceMetric(
            model_name=prev_model, cache_name=("space-" if "space" in dataset_eval else "amasum-") + self.split_str
        )
        adjusted_prevalence, (prevs, reds, trivs, gens), _ = prevmet.get_prevalence(
            [[" ".join(rev["sentences"]) for rev in row["reviews"][:review_limit]] for row in eval_data],
            predicted_summaries,
            pbar=not silent,
            product_names=[row["entity_name"] for row in eval_data],
            trivial_template=trivial_template,
            include_generics=True,
        )
        result["prevalence"] = (
            adjusted_prevalence * 100,
            (
                np.mean(prevs) * 100,
                np.mean(reds) * 100,
                np.mean(trivs) * 100,
                np.mean(gens) * 100,
            ),
        )

        return result

    def cleanup_llm_output(self, text):
        # Cleanup whitespace
        sents = sent_tokenize(text.replace("\n", " ").strip())
        # Strip "helpful" LLM padding
        sents = [
            sent.strip()
            for sent in sents
            if sent.lower()[:4] != "sure" and sent.lower()[:9] != "here is a" and sent.lower()[:8] != "here are"
        ]
        sents = [
            sent[1:-1].strip() if sent[0] == '"' and sent[-1] == '"' else sent for sent in sents
        ]  # inefficiently strip quotes
        sents = [
            sent[1:-1].strip() if sent[0] == "'" and sent[-1] == "'" else sent for sent in sents
        ]  # inefficiently strip quotes
        return " ".join(sents).strip()
