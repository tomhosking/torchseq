import os, json, jsonlines

from torchseq.eval.recipes import EvalRecipe
from torchseq.utils.model_loader import model_from_path
from torchseq.metric_hooks.opsumm_cluster_aug import OpSummClusterAugMetricHook

PROMPT_TEMPLATE_PIECEWISE = """Here is a list of sentences taken from reviews of a single hotel:

{:}

In no more than 15 words, write a single short sentence using very simple language that includes the main point:
"""

PROMPT_TEMPLATE_ONESHOT = """Here is a list of sentences taken from reviews of a single hotel:

{:}

In no more than 70 words, write a brief summary using very simple language that includes the main points:
"""


class Recipe(EvalRecipe):
    name: str = "opagg.twostage_pre"

    def run(self):
        result = {}

        if os.path.exists(os.path.join(self.model_path, "eval", f"summaries_{self.split_str}.json")):
            # Load pre generated clusters
            with open(os.path.join(self.model_path, "eval", f"summaries_{self.split_str}.json")) as f:
                summaries = json.load(f)
        else:
            # Load the model and build the clusters now
            instance = model_from_path(self.model_path, use_cuda=(not self.cpu))

            _, summaries = OpSummClusterAugMetricHook.eval_extract_summaries_and_score(
                self.config, instance, test=self.test
            )

        # Take the selected clusters and construct the prompts to hand off to an external LLM
        clusters_per_entity = summaries["evidence"]
        entity_ids = summaries["entity_ids"]

        prompts_flat_piecewise = [
            {"entity_id": ent_id, "prompt": PROMPT_TEMPLATE_PIECEWISE.format("\n".join(cluster))}
            for clusters, ent_id in zip(clusters_per_entity, entity_ids)
            for cluster in clusters
        ]

        result["prompts_piecewise"] = prompts_flat_piecewise

        with jsonlines.open(
            os.path.join(self.model_path, "eval", f"llm_inputs_piecewise_{self.split_str}.jsonl"), "w"
        ) as writer:
            writer.write_all(prompts_flat_piecewise)

        prompts_flat_oneshot = [
            {
                "entity_id": ent_id,
                "prompt": PROMPT_TEMPLATE_ONESHOT.format(
                    "\n".join([sent for cluster in clusters for sent in cluster])
                ),
            }
            for clusters, ent_id in zip(clusters_per_entity, entity_ids)
        ]

        result["prompts_oneshot"] = prompts_flat_oneshot

        with jsonlines.open(
            os.path.join(self.model_path, "eval", f"llm_inputs_oneshot_{self.split_str}.jsonl"), "w"
        ) as writer:
            writer.write_all(prompts_flat_oneshot)

        return result
