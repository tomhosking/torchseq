import numpy as np
from scipy.special import binom


def C2(x):
    return binom(x, 2)


def get_cluster_ari(predictions, references):
    scores = []
    for curr_references, curr_predictions in zip(references, predictions):
        n = len(
            set(
                [x for cluster in curr_references for x in cluster]
                + [x for cluster in curr_predictions for x in cluster]
            )
        )

        nij = np.array([[len(set(X) & set(Y)) for Y in curr_predictions] for X in curr_references])
        ai = nij.sum(axis=0)
        bj = nij.sum(axis=1)

        numerator = C2(nij).sum() - (C2(ai).sum() * C2(bj).sum()) / C2(n)

        denominator = 0.5 * (C2(ai).sum() + C2(bj).sum()) - (C2(ai).sum() * C2(bj).sum()) / C2(n)
        scores.append(numerator / denominator)
    return np.mean(scores)
