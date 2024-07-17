import os, json, jsonlines

from torchseq.eval.recipes import EvalRecipe
from torchseq.utils.model_loader import model_from_path
from torchseq.metric_hooks.opsumm_cluster_aug import OpSummClusterAugMetricHook

from nltk.tokenize import word_tokenize
import numpy as np

PROMPT_TEMPLATE_SENTENCEWISE = """Here is a list of sentences taken from reviews of the {:}:

{:}

In no more than 10 words, write a single concise sentence that includes the main point:
"""

# PROMPT_TEMPLATE_ONESHOT = """Here is a list of sentences taken from reviews of the {:}:

# {:}

# In no more than 60 words, write a concise and informative summary that includes the main points:
# """

PROMPT_TEMPLATE_ONESHOT = """Here is a list of sentences taken from reviews of the {:}:

{:}

In no more than 60 words, write a concise summary that includes the main points:
"""

PROMPT_TEMPLATE_ONESHOT_CITATIONS = """Here are some groups of sentences taken from reviews of the {:}:

{:}

In no more than 60 words, write a concise summary that includes the main points, citing which group contains the relevant information:
"""


class Recipe(EvalRecipe):
    name: str = "opagg.twostage_pre"

    cluster_limit = 24

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

        product_type = "hotel" if "space" in self.model_path else "product"
        entity_names = summaries["entity_names"]

        # Sentence-wise generation
        prompts_flat_sentencewise = [
            {
                "entity_id": ent_id,
                "prompt": PROMPT_TEMPLATE_SENTENCEWISE.format(entity_name, "\n".join(cluster[: self.cluster_limit])),
            }
            for clusters, ent_id, entity_name in zip(clusters_per_entity, entity_ids, entity_names)
            for cluster in clusters
        ]

        result["prompts_sentencewise"] = max(
            [len(word_tokenize(prompt["prompt"])) for prompt in prompts_flat_sentencewise]
        )

        with jsonlines.open(
            os.path.join(self.model_path, "eval", f"llm_inputs_piecewise_{self.split_str}.jsonl"), "w"
        ) as writer:
            writer.write_all(prompts_flat_sentencewise)


        # Document-level generation
        prompts_flat_oneshot = [
            {
                "entity_id": ent_id,
                "prompt": PROMPT_TEMPLATE_ONESHOT.format(
                    entity_name, "\n".join([sent for cluster in clusters for sent in cluster[: self.cluster_limit]])
                ),
            }
            for clusters, ent_id, entity_name in zip(clusters_per_entity, entity_ids, entity_names)
        ]

        result["prompts_oneshot"] = max([len(word_tokenize(prompt["prompt"])) for prompt in prompts_flat_oneshot])

        with jsonlines.open(
            os.path.join(self.model_path, "eval", f"llm_inputs_oneshot_{self.split_str}.jsonl"), "w"
        ) as writer:
            writer.write_all(prompts_flat_oneshot)


        # Doc-level, with citations
        prompts_flat_oneshot_citations = [
            {
                "entity_id": ent_id,
                "prompt": PROMPT_TEMPLATE_ONESHOT_CITATIONS.format(
                    entity_name, "\n\n".join(["[{:}]\n".format(cid+1) + "\n".join([sent  for sent in cluster[: self.cluster_limit]]) for cid, cluster in enumerate(clusters)])
                ),
            }
            for clusters, ent_id, entity_name in zip(clusters_per_entity, entity_ids, entity_names)
        ]

        result["prompts_oneshot_citations"] = max([len(word_tokenize(prompt["prompt"])) for prompt in prompts_flat_oneshot_citations])

        with jsonlines.open(
            os.path.join(self.model_path, "eval", f"llm_inputs_oneshot_citations_{self.split_str}.jsonl"), "w"
        ) as writer:
            writer.write_all(prompts_flat_oneshot_citations)

        return result
