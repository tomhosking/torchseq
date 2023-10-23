from typing import Literal, Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveTripletLoss(nn.Module):
    metric: Literal["euclidean", "cosine", "dot"]
    loss_type: Literal["softnn", "marginmse"]
    tau: float

    def __init__(
        self,
        metric: Literal["euclidean", "cosine", "dot"] = "euclidean",
        loss_type: Literal["softnn", "marginmse"] = "softnn",
        tau: float = 1.0,
    ):
        super(ContrastiveTripletLoss, self).__init__()

        if metric not in ["euclidean", "cosine", "dot"]:
            raise Exception("Unsupported metric in ContrastiveLoss: {:}".format(metric))

        self.metric = metric
        self.loss_type = loss_type
        self.tau = tau

    def forward(
        self,
        query_encodings: torch.Tensor,
        pos_encodings: torch.Tensor,
        neg_encodings: torch.Tensor,
        pos_scores: Optional[torch.Tensor] = None,
        neg_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pos_scores is not None:
            assert (
                query_encodings.shape[0] == pos_scores.shape[0]
            ), "pos_scores should have shape bsz x 1 in ContrastiveTripletLoss! Found {:}".format(pos_scores.shape)
        else:
            pos_scores = torch.tensor(1.0)

        if neg_scores is not None:
            assert (
                query_encodings.shape[0] == neg_scores.shape[0]
            ), "neg_scores should have shape bsz x 1 in ContrastiveTripletLoss! Found {:}".format(neg_scores.shape)
        else:
            neg_scores = torch.tensor(1e-18)

        if self.metric == "euclidean":
            pos_distances = (query_encodings - pos_encodings) ** 2
            pos_distances = pos_distances.sum(dim=-1)  # remove data dim
            pos_distances = torch.exp(-pos_distances / self.tau)
            neg_distances = (query_encodings - neg_encodings) ** 2
            neg_distances = neg_distances.sum(dim=-1)  # remove data dim
            neg_distances = torch.exp(-neg_distances / self.tau)
        elif self.metric == "cosine":
            pos_distances = -1 * F.cosine_similarity(query_encodings, pos_encodings, dim=-1)
            neg_distances = -1 * F.cosine_similarity(query_encodings, neg_encodings, dim=-1)
        elif self.metric == "dot":
            pos_distances = -1 * (query_encodings * pos_encodings).sum(dim=-1)
            neg_distances = -1 * (query_encodings * neg_encodings).sum(dim=-1)
        else:
            raise Exception("Unsupported metric in ContrastiveLoss: {:}".format(self.metric))

        if self.loss_type == "softnn":
            numerator = torch.log(pos_distances.squeeze(1) * pos_scores + neg_distances.squeeze(1) * neg_scores)
            denom = torch.log(pos_distances.squeeze(1) + neg_distances.squeeze(1))

            # if pos_scores is not None:
            #     numerator = torch.logsumexp(
            #         logits + (scores + 1e-18).log() - eye_mask.logical_not() * 1e18, dim=-1, keepdim=True
            #     )
            # else:
            #     numerator = torch.logsumexp(logits - pos_mask.logical_not() * 1e18, dim=-1, keepdim=True)
            # denom = torch.logsumexp(logits - eye_mask.logical_not() * 1e18, dim=-1, keepdim=True)

            loss = -1.0 * (numerator - denom)

        elif self.loss_type == "marginmse":
            pred_margin = pos_distances - neg_distances
            tgt_margin = pos_scores - neg_scores
            # print(pred_margin.shape, tgt_margin.shape)
            loss = (pred_margin.squeeze(1) - tgt_margin) ** 2
        else:
            raise Exception("Unsupported loss type in ContrastiveLoss: {:}".format(self.metric))

        return loss
