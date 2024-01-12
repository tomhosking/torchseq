import os, json, jsonlines

from torchseq.eval.recipes import EvalRecipe
from torchseq.utils.model_loader import model_from_path
from torchseq.metric_hooks.self_retrieval import SelfRetrievalMetricHook

PROMPT_TEMPLATE = """Here is a list of sentences taken from hotel reviews:

{:}

In no more than 15 words, write a single short sentence using very simple language that includes the main point:
"""


class OpAggTwoStagePostEvalRecipe(EvalRecipe):
    def run(self):
        result = {}

        if os.path.exists(os.path.join(self.model_path, "eval", f"summaries_{self.split_str}.json")):
            # Load pre generated clusters
            with open(os.path.join(self.model_path, "eval", f"summaries_{self.split_str}.json")) as f:
                summaries = json.load(f)
        else:
            # Load the model and build the clusters now
            instance = model_from_path(self.model_path, use_cuda=(not self.cpu))

            _, summaries = SelfRetrievalMetricHook.eval_extract_summaries_and_score(
                self.config, instance, test=self.test
            )

        # Take the selected clusters and construct the prompts to hand off to an external LLM
        clusters_per_entity = summaries["evidence"]

        # TODO: preprocess clusters here? remove dupes, combine similar etc

        # TODO: include metadata
        prompts_flat = [
            {"prompt": PROMPT_TEMPLATE.format("\n".join(cluster))}
            for clusters in clusters_per_entity
            for cluster in clusters
        ]

        result["prompts"] = prompts_flat

        with jsonlines.open(os.path.join(self.model_path, "eval", f"llm_inputs_{self.split_str}.jsonl")) as writer:
            writer.write_all(prompts_flat)

        return result
