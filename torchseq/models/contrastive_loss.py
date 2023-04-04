from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    metric: Literal["euclidean", "cosine"]
    loss_type: Literal["softnn", "basic"]

    def __init__(
        self, metric: Literal["euclidean", "cosine"] = "euclidean", loss_type: Literal["softnn", "basic"] = "softnn"
    ):
        super(ContrastiveLoss, self).__init__()

        if metric not in ["euclidean", "cosine"]:
            raise Exception("Unsupported metric in ContrastiveLoss: {:}".format(metric))

        self.metric = metric
        self.loss_type = loss_type

    def forward(self, encodings: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        # Get a mask that's true for other members of the same group
        eye_mask = torch.eye(encodings.shape[0], dtype=torch.bool).logical_not().to(encodings.device)
        pos_mask = (groups.unsqueeze(0) == groups.unsqueeze(1)).logical_and(eye_mask)
        neg_mask = groups.unsqueeze(0) != groups.unsqueeze(1)

        distances = torch.zeros(encodings.shape[0])
        if self.metric == "euclidean":
            distances = (encodings - encodings.transpose(0, 1)) ** 2
            distances = distances.sum(dim=-1)  # remove data dim
        elif self.metric == "cosine":
            distances = -1 * F.cosine_similarity(encodings, encodings.transpose(0, 1), dim=-1)
        else:
            raise Exception("Unsupported metric in ContrastiveLoss: {:}".format(self.metric))

        if self.loss_type == "softnn":
            tau = 1.0
            # print(distances.shape)
            pos_loss = torch.logsumexp(-distances * pos_mask / tau, dim=-1, keepdim=True)
            # neg_loss = torch.logsumexp(-distances * neg_mask / tau, dim=-1, keepdim=True)
            # denom = torch.logsumexp(-distances * eye_mask / tau, dim=-1, keepdim=True)

            # print(pos_loss.shape)
            # print(neg_loss.shape)
            # print(denom.shape)
            # loss = -(torch.logsumexp(torch.stack([pos_loss, neg_loss], dim=-1),dim=-1) - denom).sum(-1)
            # loss = -(pos_loss - denom).sum(-1)

            logits = -distances / tau - torch.eye(distances.shape[0], device=distances.device) * 1e18
            log_probs = F.log_softmax(logits, dim=-1) * pos_mask

            loss = (-1.0 * log_probs).sum(dim=-1)
            # print(logits.shape)
            # print(log_probs.shape)
            # print(loss.shape)
            # print(logits)
            # print(log_probs)
            # print(loss)
            # exit()

        elif self.loss_type == "basic":
            pos_loss = (distances * pos_mask).sum(dim=-1) / (pos_mask.sum(dim=-1) + 1e-10)
            neg_loss = (distances * neg_mask).sum(dim=-1) / (neg_mask.sum(dim=-1) + 1e-10)
            regularizer = (
                distances.diagonal(dim1=-1, dim2=-2) if self.metric == "euclidean" else torch.zeros_like(pos_loss)
            )

            loss = pos_loss - neg_loss + regularizer
        else:
            raise Exception("Unsupported loss type in ContrastiveLoss: {:}".format(self.metric))

        return loss
