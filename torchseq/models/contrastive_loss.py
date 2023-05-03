from typing import Literal, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchseq.pretrained.nli import PretrainedNliModel

from nltk.corpus import stopwords


class ContrastiveLoss(nn.Module):
    metric: Literal["euclidean", "cosine", "dot"]
    loss_type: Literal["softnn", "basic"]
    tau: float

    def __init__(
        self,
        metric: Literal["euclidean", "cosine", "dot"] = "euclidean",
        loss_type: Literal["softnn", "basic"] = "softnn",
        tau: float = 1.0,
    ):
        super(ContrastiveLoss, self).__init__()

        if metric not in ["euclidean", "cosine", "dot"]:
            raise Exception("Unsupported metric in ContrastiveLoss: {:}".format(metric))

        self.metric = metric
        self.loss_type = loss_type
        self.tau = tau

    def forward(
        self,
        encodings: torch.Tensor,
        encodings2: Optional[torch.Tensor] = None,
        groups: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if groups is None and scores is None and encodings2 is None:
            raise Exception("At least one of `group`, `weights` or `encodings2` must be passed to ContrastiveLoss!")

        if scores is not None:
            assert (
                encodings.shape[0] == scores.shape[0]
            ), "scores should have shape bsz x bsz in ContrastiveLoss! Found {:}".format(scores.shape)
            assert (
                encodings.shape[0] == scores.shape[1]
            ), "scores should have shape bsz x bsz in ContrastiveLoss! Found {:}".format(scores.shape)

        # Don't include interaction terms between the same elements!
        eye_mask = torch.eye(encodings.shape[0], dtype=torch.bool).logical_not().to(encodings.device)

        if groups is not None:
            # Get a mask that's true for other members of the same group
            pos_mask = (groups.unsqueeze(0) == groups.unsqueeze(1)).logical_and(eye_mask)
            neg_mask = groups.unsqueeze(0) != groups.unsqueeze(1)
        elif encodings2 is not None:
            pos_mask = eye_mask.logical_not()
            neg_mask = torch.ones_like(pos_mask, dtype=torch.bool)
        else:
            pos_mask = eye_mask
            neg_mask = eye_mask

        distances = torch.zeros(encodings.shape[0])

        # Allow for
        if encodings2 is None:
            encodings2 = encodings

        if self.metric == "euclidean":
            distances = (encodings - encodings2.transpose(0, 1)) ** 2
            distances = distances.sum(dim=-1)  # remove data dim
        elif self.metric == "cosine":
            distances = -1 * F.cosine_similarity(encodings, encodings2.transpose(0, 1), dim=-1)
        elif self.metric == "dot":
            distances = -1 * (encodings * encodings2.transpose(0, 1)).sum(dim=-1)
        else:
            raise Exception("Unsupported metric in ContrastiveLoss: {:}".format(self.metric))

        if self.loss_type == "softnn":
            logits = -distances / self.tau

            if scores is not None:
                numerator = torch.logsumexp(
                    logits + (scores + 1e-18).log() - eye_mask.logical_not() * 1e18, dim=-1, keepdim=True
                )
            else:
                numerator = torch.logsumexp(logits - pos_mask.logical_not() * 1e18, dim=-1, keepdim=True)
            denom = torch.logsumexp(logits - eye_mask.logical_not() * 1e18, dim=-1, keepdim=True)

            loss = (-1.0 * (numerator - denom)).sum(dim=-1)

            if scores is not None:
                loss = torch.where(scores.sum(dim=-1) > 1e-5, loss, torch.zeros_like(loss))

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


class NliContrastiveLoss(nn.Module):
    cont_loss: ContrastiveLoss
    nli_model: PretrainedNliModel
    src_field: str
    bsz: int

    def __init__(
        self,
        metric: Literal["euclidean", "cosine", "dot"] = "euclidean",
        loss_type: Literal["softnn", "basic"] = "softnn",
        tau: float = 1.0,
        src_field: str = "s1",
        bsz: int = 64,
    ):
        super(NliContrastiveLoss, self).__init__()
        self.cont_loss = ContrastiveLoss(metric, loss_type, tau)
        self.nli_model = PretrainedNliModel()
        self.src_field = src_field
        self.bsz = bsz

    def forward(self, encodings: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Calculate pairwise scores

        # input_sents = batch[self.src_field + "_text"]
        # input_groups = batch[self.src_field + "_group"]
        input_clusters = batch[self.src_field + "_clusters_text"]
        cluster_sizes = [len(cluster) for cluster in input_clusters]

        def remove_punctuation(test_str):
            # Using filter() and lambda function to filter out punctuation characters
            result = "".join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), test_str))
            return result

        def get_bow_sim(x, y):
            x_bag = set(remove_punctuation(x).split()) - set(stopwords.words("english"))
            y_bag = set(remove_punctuation(y).split()) - set(stopwords.words("english"))
            return len(x_bag & y_bag) / len(x_bag)

        BOW_SIM_THRESHOLD = 0.2

        bow_sims = {
            (s1, s2): get_bow_sim(s1, s2) for cluster in input_clusters for s1 in cluster for s2 in cluster if s1 != s2
        }
        intra_pairs = [
            (s1, s2)
            for cluster in input_clusters
            for s1 in cluster
            for s2 in cluster
            if s1 != s2 and bow_sims[(s1, s2)] > BOW_SIM_THRESHOLD
        ]
        intra_premises, intra_hypotheses = zip(*intra_pairs)

        nli_scores_intra_flat = self.nli_model.get_scores(
            premises=intra_premises, hypotheses=intra_hypotheses, bsz=self.bsz
        )

        i = 0
        nli_scores_intra = []
        for cluster in input_clusters:
            cluster_scores = torch.zeros(len(cluster), len(cluster))
            for j, s1 in enumerate(cluster):
                for k, s2 in enumerate(cluster):
                    if s1 != s2 and bow_sims[(s1, s2)] > BOW_SIM_THRESHOLD:
                        cluster_scores[j, k] = nli_scores_intra_flat[i]
                        i += 1

            nli_scores_intra.append(cluster_scores)

        # print(nli_scores_intra)
        # exit()

        centroids = [torch.argmax(cluster.sum(dim=-1)).item() for cluster in nli_scores_intra]
        # print(centroids)

        inter_pairs = [
            (c1[i], c2[j])
            for c1, i in zip(input_clusters, centroids)
            for c2, j in zip(input_clusters, centroids)
            if c1[i] != c2[j]
        ]
        inter_premises, inter_hypotheses = zip(*inter_pairs)

        nli_scores_inter_flat = self.nli_model.get_scores(
            premises=inter_premises, hypotheses=inter_hypotheses, bsz=self.bsz
        )

        # print(len(nli_scores_flat))

        nli_scores_inter = torch.zeros(len(input_clusters), len(input_clusters))
        i = 0
        for j, (cluster1, cent1) in enumerate(zip(input_clusters, centroids)):
            for k, (cluster2, cent2) in enumerate(zip(input_clusters, centroids)):
                if cluster1[cent1] != cluster2[cent2]:
                    nli_scores_inter[j, k] = nli_scores_inter_flat[i]
                    i += 1

        # Construct the full grid of inter-cluster scores
        nli_scores_inter = torch.repeat_interleave(nli_scores_inter, torch.tensor(cluster_sizes), dim=0)
        nli_scores_inter = torch.repeat_interleave(nli_scores_inter, torch.tensor(cluster_sizes), dim=1)

        nli_scores_intra_expanded = torch.block_diag(*nli_scores_intra)
        # print(nli_scores_intra_expanded.shape)

        nli_scores = nli_scores_inter + nli_scores_intra_expanded

        # Threshold the scores to be at least 0.1
        nli_scores = torch.where(nli_scores > 0.1, nli_scores, torch.zeros_like(nli_scores))

        # print(nli_scores)
        # print(nli_scores.shape)
        # print(encodings.shape)
        # print(len(intra_hypotheses))
        # print(len(inter_hypotheses))
        # exit()

        # TODO: Check whether this is the right way round!!!

        # Calculate contrastive loss
        loss: torch.Tensor = self.cont_loss(encodings, scores=nli_scores.to(encodings.device))
        return loss
