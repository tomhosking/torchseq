# Modified version of the metric from https://github.com/cdmalon/opinion-prevalence
# Original is a CLI, this provides a more generic OO wrapper
# See https://arxiv.org/abs/2307.14305

import os
import nltk.tokenize
from summac.model_summac import SummaCZS
from tqdm import tqdm
import numpy as np
from typing import List, Union, Optional, Literal, Sequence


class PrevalenceMetric:
    threshold: float = 0.04
    model: SummaCZS
    use_cache: bool

    def __init__(
        self,
        model_name: Literal["vitc", "vitc-base", "mnli", "mnli-base"] = "mnli",
        threshold: float = 0.04,
        use_cache: bool = True,
    ):
        # Default model was originally mnli - changed to vitc
        self.model = SummaCZS(
            granularity="document", model_name=model_name, bins="percentile", use_con=False, device="cuda"
        )
        self.threshold = threshold

        self.use_cache = use_cache

        if self.use_cache:
            self.model.imager.cache_folder = os.path.expanduser("~/.summac_cache/")
            os.makedirs(self.model.imager.cache_folder, exist_ok=True)
            cache_file = self.model.imager.get_cache_file()
            self.model.imager.load_cache()

    def get_prevalence(
        self,
        reviews: List[List[str]],
        generated_summaries: Sequence[Union[str, List[str]]],
        product_names: Optional[List[str]] = None,
        pbar: bool = False,
        ignore_redundancy: bool = False,
        summaries_are_sentences: bool = False,
        trivial_template: str = "I stayed at {:}.",
        trivial_default: str = "a hotel",
        batch_size: int = 32,
    ):
        if product_names is None:
            product_names = [trivial_default] * len(reviews)

        prevalences = []
        redundancies = []
        trivials = []
        for curr_reviews, summ, product_name in tqdm(
            zip(reviews, generated_summaries, product_names), disable=(not pbar), total=len(reviews)
        ):
            if not summaries_are_sentences:
                sents = nltk.tokenize.sent_tokenize(summ)
            else:
                sents = summ

            # Prepare inputs
            trivial = trivial_template.format(product_name)

            trivial_inputs = [trivial] * len(sents), sents

            if not ignore_redundancy:
                redundant_inputs = [sents[j] for i, _ in enumerate(sents) for j in range(i)], [
                    sent for i, sent in enumerate(sents) for _ in range(i)
                ]
            else:
                redundant_inputs = [], []

            implied_inputs = [curr_reviews[k] for k in range(len(curr_reviews)) for _ in sents], sents * len(
                curr_reviews
            )

            all_inputs = (
                trivial_inputs[0] + redundant_inputs[0] + implied_inputs[0],
                trivial_inputs[1] + redundant_inputs[1] + implied_inputs[1],
            )

            trivial_offset = len(trivial_inputs[0])
            redundant_offset = len(trivial_inputs[0]) + len(redundant_inputs[0])

            # Get all SummaC scores for this summary
            all_scores = self.model.score(*all_inputs, batch_size=batch_size)["scores"]

            # Calculate which summary sentences were trivial
            trivial_scores = all_scores[:trivial_offset]
            trivial_mask = np.array(trivial_scores) > self.threshold

            # Calculate which sentences are redundant (wrt previous sentences)
            if not ignore_redundancy:
                redundant_scores = all_scores[trivial_offset:redundant_offset]
                redundant_mask_flat = np.array(redundant_scores) > self.threshold
                redundant_mask_list = []
                k = 0
                for i, sent in enumerate(sents):
                    row = []
                    for j in range(i):
                        row.append(redundant_mask_flat[k])
                        k += 1
                    redundant_mask_list.append(np.array(row).any())

                redundant_mask = np.array(redundant_mask_list)
                redundant_mask = redundant_mask & np.logical_not(trivial_mask)  # Ignore if already marked as trivial
            else:
                redundant_mask = np.zeros_like(trivial_mask)

            # Calculate which sentences are supported by reviews
            implied_scores = all_scores[redundant_offset:]
            implied_mask_flat = np.array(implied_scores) > self.threshold
            implied_counts = implied_mask_flat.reshape(len(curr_reviews), len(sents)).mean(axis=0)

            implied_counts = implied_counts * (
                np.logical_not(trivial_mask) & (ignore_redundancy | np.logical_not(redundant_mask))
            )  # Ignore if trivial or redundant

            # Aggregate
            prevalences.append(implied_counts)
            redundancies.append(redundant_mask)
            trivials.append(trivial_mask)

        if self.use_cache:
            self.model.imager.save_cache()

        return (
            [np.mean(prevs) for prevs in prevalences],
            [np.mean(reds) for reds in redundancies],
            [np.mean(trivs) for trivs in trivials],
        ), (
            prevalences,
            redundancies,
            trivials,
        )
