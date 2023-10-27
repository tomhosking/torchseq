from typing import Literal, Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveTripletLoss(nn.Module):
    metric: Literal["euclidean", "cosine", "dot"]
    loss_type: Literal["softnn", "marginmse"]
    tau: float
    inbatch_negatives: bool

    def __init__(
        self,
        metric: Literal["euclidean", "cosine", "dot"] = "euclidean",
        loss_type: Literal["softnn", "marginmse"] = "softnn",
        tau: float = 1.0,
        inbatch_negatives: bool = False,
        softnn_agg_mean: bool = False,
    ):
        super(ContrastiveTripletLoss, self).__init__()

        if metric not in ["euclidean", "cosine", "dot"]:
            raise Exception("Unsupported metric in ContrastiveLoss: {:}".format(metric))

        self.metric = metric
        self.loss_type = loss_type
        self.tau = tau
        self.inbatch_negatives = inbatch_negatives
        self.softnn_agg_mean = softnn_agg_mean

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
            neg_distances = (query_encodings - neg_encodings.transpose(0, 1)) ** 2
            neg_distances = neg_distances.sum(dim=-1)  # remove data dim
            neg_distances = torch.exp(-neg_distances / self.tau)
        elif self.metric == "cosine":
            pos_distances = torch.exp(-1 * F.cosine_similarity(query_encodings, pos_encodings, dim=-1))
            neg_distances = torch.exp(-1 * F.cosine_similarity(query_encodings, neg_encodings.transpose(0, 1), dim=-1))
        elif self.metric == "dot":
            pos_distances = torch.exp(-1 * (query_encodings * pos_encodings).sum(dim=-1))
            neg_distances = torch.exp(-1 * (query_encodings * neg_encodings.transpose(0, 1)).sum(dim=-1))
            # exit()
        else:
            raise Exception("Unsupported metric in ContrastiveLoss: {:}".format(self.metric))

        if self.loss_type == "softnn":
            if self.inbatch_negatives:
                # Assume scores for in-batch negs are 0 - then they only need to be included in the denominator
                numerator = torch.log(pos_distances.squeeze(1) * pos_scores + neg_distances.diag() * neg_scores)

                denom = torch.log(
                    pos_distances.squeeze(1) + neg_distances.mean(dim=1)
                    if self.softnn_agg_mean
                    else neg_distances.sum(dim=1)
                )
            else:
                numerator = torch.log(pos_distances.squeeze(1) * pos_scores + neg_distances.diag() * neg_scores)
                denom = torch.log(pos_distances.squeeze(1) + neg_distances.diag())

            # if pos_scores is not None:
            #     numerator = torch.logsumexp(
            #         logits + (scores + 1e-18).log() - eye_mask.logical_not() * 1e18, dim=-1, keepdim=True
            #     )
            # else:
            #     numerator = torch.logsumexp(logits - pos_mask.logical_not() * 1e18, dim=-1, keepdim=True)
            # denom = torch.logsumexp(logits - eye_mask.logical_not() * 1e18, dim=-1, keepdim=True)

            loss = -1.0 * (numerator - denom)

        elif self.loss_type == "marginmse":
            if self.inbatch_negatives:
                # raise Exception('In batch negatives are not yet supported for MarginMSE loss')

                # Pos distances is bsz, neg_distances is bsz x bsz
                # Pos scores is bsz, neg_scores is ALSO bsz
                # So scatter neg_scores along diag

                # print('## ContTripLoss')
                # print(pos_distances.shape, neg_distances.shape)
                # print(neg_scores.shape)

                neg_scores_expanded = torch.zeros_like(neg_distances)
                neg_scores = torch.diagonal_scatter(neg_scores_expanded, neg_scores, 0)
                pred_margin = pos_distances - neg_distances

                # Scores are SIMILARITIES not distances! So need to take negative
                tgt_margin = -pos_scores.unsqueeze(1) + neg_scores

                loss = ((pred_margin.squeeze(1) - tgt_margin) ** 2).mean(1)
                # print(loss.shape)
            else:
                pred_margin = pos_distances - neg_distances
                # Scores are SIMILARITIES not distances! So need to take negative
                tgt_margin = -pos_scores + neg_scores
                # print(pred_margin.shape, tgt_margin.shape)
                loss = (pred_margin.squeeze(1) - tgt_margin) ** 2
        else:
            raise Exception("Unsupported loss type in ContrastiveLoss: {:}".format(self.metric))

        return loss
