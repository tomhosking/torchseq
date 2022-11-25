#!/usr/bin/env python3


import numpy as np
from rouge_score import rouge_scorer, scoring

# Based on the GEM ROUGE implementation: https://github.com/GEM-benchmark/GEM-metrics/blob/main/gem_metrics/rouge.py
"""ROUGE uses Google implementation (https://github.com/google-research/google-research/tree/master/rouge)
but adds own implementation of multi-ref jackknifing.
The Google implementation should be identical to Rouge-155 (except tokenization?),
the jackknifing follows the description of the ROUGE paper.
"""


def get_pairwise_rouge(pred, ref):
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
    res = rouge.score(ref, pred)
    return {rtype: res[rtype].fmeasure for rtype in rouge_types}


def get_jackknife_rouge(predictions, references, stemming=True):
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=stemming)
    score_list = []

    for refs, pred in zip(
        references,
        predictions,
    ):
        # ROUGE multi-ref jackknifing
        if len(refs) > 1:
            cur_scores = [rouge.score(ref, pred) for ref in refs]

            # get best score for all leave-one-out sets
            best_scores = []
            for leave in range(len(refs)):
                cur_scores_leave_one = [cur_scores[s] for s in range(len(refs)) if s != leave]
                best_scores.append(
                    {
                        rouge_type: max(
                            [s[rouge_type] for s in cur_scores_leave_one],
                            key=lambda s: s.fmeasure,
                        )
                        for rouge_type in rouge_types
                    }
                )

            # average the leave-one-out bests to produce the final score
            score = {
                rouge_type: scoring.Score(
                    np.mean([b[rouge_type].precision for b in best_scores]),
                    np.mean([b[rouge_type].recall for b in best_scores]),
                    np.mean([b[rouge_type].fmeasure for b in best_scores]),
                )
                for rouge_type in rouge_types
            }
        else:
            score = rouge.score(refs[0], pred)

        # convert the named tuples to plain nested dicts
        score = {
            rouge_type: {
                "precision": score[rouge_type].precision,
                "recall": score[rouge_type].recall,
                "fmeasure": score[rouge_type].fmeasure,
            }
            for rouge_type in rouge_types
        }
        score_list.append(score)

    l1_keys = list(score_list[0].keys())
    # l2_keys = score_list[0][l1_keys[0]].keys()
    return {key1: round(np.mean([score[key1]["fmeasure"] for score in score_list]), 5) for key1 in l1_keys}
