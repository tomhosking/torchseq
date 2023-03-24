import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    metric: str

    def __init__(self, metric: str = "euclidean"):
        super(ContrastiveLoss, self).__init__()

        if metric not in ["euclidean", "cosine"]:
            raise Exception("Unsupported metric in ContrastiveLoss: {:}".format(metric))

        self.metric = metric

    def forward(self, encodings: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        # Get a mask that's true for other members of the same group
        group_mask = groups.unsqueeze(0) == groups.unsqueeze(1)
        group_mask = group_mask.logical_and(
            torch.eye(group_mask.shape[0], dtype=torch.bool).logical_not().to(encodings.device)
        )

        loss = torch.zeros(encodings.shape[0])
        if self.metric == "euclidean":
            loss = (encodings - encodings.transpose(0, 1)) ** 2
            loss = loss.mean(dim=-1) # remove data dim
        elif self.metric == "cosine":
            raise Exception("Cosine metric not yet implemented")
        else:
            raise Exception("Unsupported metric in ContrastiveLoss: {:}".format(self.metric))

        # TODO: check reduction here - should probs evenly weight +ve and -ve, and work out actual counts of each
        loss = -1 * loss * group_mask + loss * (group_mask.logical_not())
        loss = loss.mean(dim=-1) # remove "other batch" dim

        return loss
