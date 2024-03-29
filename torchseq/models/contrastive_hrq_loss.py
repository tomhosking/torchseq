from typing import Literal, Optional, Dict, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchseq.utils.functions import get_off_diagonal_elements, cos_sim

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from math import exp


class HierarchicalQuantizationLoss(nn.Module):
    tau: float

    def __init__(
        self,
        similarity_fn: Literal["crossent", "kl", "euclidean", "dot", "cosine"] = "crossent",
        tau: float = 1.0,
        gamma: Optional[float] = 2.0,
        lamda: float = 2.0,
        inbatch_negatives: bool = False,
        softnn_agg_mean: bool = False,
        inbatch_weight: Optional[float] = None,
        inbatch_weight_decaysteps: Optional[int] = None,
        inbatch_weight_min: Optional[float] = None,
        tgt_nograd: bool = False,
        hierarchy_mask: bool = False,
        hierarchy_mask_smoothing: Optional[float] = None,
        hierarchy_weight_dim: int = 1,
        query_use_probs: bool = False,
        tgt_use_probs: bool = False,
        maximise_expected_scores: bool = False,
        detach_prev_levels: bool = False,
        query_leaves_only: bool = False,
        tgt_leaves_only: bool = False,
        mean_weighting: bool = False,
        hard_neg_weight: float = 1,
        neg_topic_threshold: Optional[float] = None,
        neg_topic_use_scores: bool = False,
        topic_dataset_sents: Optional[List[str]] = None,
    ):
        super(HierarchicalQuantizationLoss, self).__init__()

        # self.loss_type = loss_type
        self.similarity_fn = similarity_fn
        self.tau = tau
        self.gamma = gamma
        self.lamda = lamda
        self.inbatch_negatives = inbatch_negatives
        self.softnn_agg_mean = softnn_agg_mean
        self.inbatch_weight = inbatch_weight
        self.inbatch_weight_decaysteps = inbatch_weight_decaysteps
        self.inbatch_weight_min = inbatch_weight_min
        self.tgt_nograd = tgt_nograd
        self.hierarchy_mask = hierarchy_mask
        self.hierarchy_mask_smoothing = hierarchy_mask_smoothing
        self.hierarchy_weight_dim = hierarchy_weight_dim
        self.query_use_probs = query_use_probs
        self.tgt_use_probs = tgt_use_probs
        self.maximise_expected_scores = maximise_expected_scores
        self.detach_prev_levels = detach_prev_levels
        self.query_leaves_only = query_leaves_only
        self.tgt_leaves_only = tgt_leaves_only
        self.mean_weighting = mean_weighting
        self.hard_neg_weight = hard_neg_weight
        self.neg_topic_threshold = neg_topic_threshold
        self.neg_topic_use_scores = neg_topic_use_scores

        if neg_topic_threshold is not None or neg_topic_use_scores:
            if topic_dataset_sents is None:
                raise Exception("If neg_topic_threshold is not null, then topic_dataset_sents needs to be set!")

            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words=("english"),
                min_df=5,
                max_df=0.5,
            )
            self.tfidf_vectorizer.fit(topic_dataset_sents)

    def forward(
        self,
        global_step: int,
        query_path_onehot: torch.Tensor,
        query_path_logits: torch.Tensor,
        query_path_embedded: torch.Tensor,
        query_text: List[str],
        pos_path_onehot: torch.Tensor,
        pos_path_logits: torch.Tensor,
        pos_path_embedded: torch.Tensor,
        pos_text: List[str],
        neg_path_onehot: torch.Tensor,
        neg_path_logits: torch.Tensor,
        neg_path_embedded: torch.Tensor,
        neg_text: List[str],
        pos_scores: Optional[torch.Tensor] = None,
        neg_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        EPS = 1e-18

        if pos_scores is None:
            pos_scores = torch.ones_like(query_path_onehot[:, 0, 0])
        if neg_scores is None:
            neg_scores = torch.zeros_like(query_path_onehot[:, 0, 0])

        if self.neg_topic_threshold is not None or self.neg_topic_use_scores:
            query_tfidf = self.tfidf_vectorizer.transform(query_text)
            neg_tfidf = self.tfidf_vectorizer.transform(neg_text)
            query_neg_tfidf_distances = torch.tensor(cosine_distances(query_tfidf, neg_tfidf)).to(
                query_path_embedded.device
            )

            if self.neg_topic_use_scores:
                query_neg_tfidf_mask = query_neg_tfidf_distances
            else:
                query_neg_tfidf_mask = torch.where(
                    query_neg_tfidf_distances < self.neg_topic_threshold,
                    torch.zeros_like(query_neg_tfidf_distances),
                    torch.ones_like(query_neg_tfidf_distances),
                )

        else:
            query_neg_tfidf_mask = torch.tensor(1.0)

        if self.detach_prev_levels:
            query_path_embedded_prev = torch.cumsum(query_path_embedded, dim=1).detach()
            pos_path_embedded_prev = torch.cumsum(pos_path_embedded, dim=1).detach()
            neg_path_embedded_prev = torch.cumsum(neg_path_embedded, dim=1).detach()

            query_path_embedded_prev = torch.cat(
                [torch.zeros_like(query_path_embedded_prev[:, :1, :]), query_path_embedded_prev[:, 1:, :]], dim=1
            )
            pos_path_embedded_prev = torch.cat(
                [torch.zeros_like(pos_path_embedded_prev[:, :1, :]), pos_path_embedded_prev[:, 1:, :]], dim=1
            )
            neg_path_embedded_prev = torch.cat(
                [torch.zeros_like(neg_path_embedded_prev[:, :1, :]), neg_path_embedded_prev[:, 1:, :]], dim=1
            )

            if self.query_leaves_only:
                # Disable detaching for queries, since we're only going to use the leaves anyway
                query_path_embedded = torch.cumsum(query_path_embedded, dim=1)
            else:
                query_path_embedded = query_path_embedded + query_path_embedded_prev

            if self.tgt_leaves_only:
                # Disable detaching for tgts, since we're only going to use the leaves anyway
                pos_path_embedded = torch.cumsum(pos_path_embedded, dim=1)
                neg_path_embedded = torch.cumsum(neg_path_embedded, dim=1)
            else:
                pos_path_embedded = pos_path_embedded + pos_path_embedded_prev
                neg_path_embedded = neg_path_embedded + neg_path_embedded_prev
        else:
            query_path_embedded = torch.cumsum(query_path_embedded, dim=1)
            pos_path_embedded = torch.cumsum(pos_path_embedded, dim=1)
            neg_path_embedded = torch.cumsum(neg_path_embedded, dim=1)

        # Calculate 1/gamma^d, then normalise
        if self.gamma is not None:
            hierarchy_weight = (torch.ones_like(query_path_onehot[:, :, 0]) / self.gamma).cumprod(dim=-1) * self.gamma
            hierarchy_weight = hierarchy_weight / hierarchy_weight.sum(-1, keepdim=True)
        else:
            hierarchy_weight = torch.ones_like(query_path_onehot[:1, 0, 0])

        # Mask each level based on whether the *previous* levels matched
        if self.hierarchy_mask:
            mask = (pos_path_onehot * query_path_onehot).sum(dim=-1, keepdim=False).cumprod(dim=1)
            mask = torch.cat([torch.ones_like(mask)[:, :1], mask[:, :-1]], dim=1)

            if self.hierarchy_mask_smoothing is not None:
                mask = mask * (1 - self.hierarchy_mask_smoothing) + self.hierarchy_mask_smoothing * torch.ones_like(
                    mask
                )

            mask = torch.pow(1 / self.lamda, mask)
        else:
            mask = torch.ones_like(pos_path_onehot[:, :1, 0])

        query_path_logprobs = (
            nn.functional.log_softmax(query_path_logits, dim=-1)
            if self.query_use_probs
            else torch.log(query_path_onehot + EPS)
        )

        pos_path_probs = nn.functional.softmax(pos_path_logits, dim=-1) if self.tgt_use_probs else pos_path_onehot
        neg_path_probs = nn.functional.softmax(neg_path_logits, dim=-1) if self.tgt_use_probs else neg_path_onehot

        if self.similarity_fn == "crossent":
            # Calculate xentropy between targets and queries
            pos_distances = -(pos_path_probs * query_path_logprobs).sum(dim=-1)  # bsz x depth
            neg_distances = -(neg_path_probs.unsqueeze(0) * query_path_logprobs.unsqueeze(1)).sum(
                dim=-1
            )  # bsz x bsz x depth
        elif self.similarity_fn == "kl":
            pos_distances = nn.functional.kl_div(
                query_path_logprobs, pos_path_probs, log_target=False, reduction="none"
            ).sum(-1)
            # pos_distances = -(pos_path_probs * (query_path_logprobs - (pos_path_probs + EPS).log())).sum(
            #     dim=-1
            # )  # bsz x depth
            neg_distances = nn.functional.kl_div(
                query_path_logprobs.unsqueeze(1), neg_path_probs.unsqueeze(0), log_target=False, reduction="none"
            ).sum(-1)
            # neg_distances = -(
            #     neg_path_probs.unsqueeze(0) * (query_path_logprobs - (neg_path_probs + EPS).log()).unsqueeze(1)
            # ).sum(
            #     dim=-1
            # )  # bsz x bsz x depth
        elif self.similarity_fn == "euclidean":
            if self.query_leaves_only and self.tgt_leaves_only:
                pos_distances = ((query_path_embedded[:, -1:, :] - pos_path_embedded[:, -1:, :]) ** 2).sum(-1)
                neg_distances = (
                    (query_path_embedded[:, -1:, :].unsqueeze(1) - neg_path_embedded[:, -1:, :].unsqueeze(0)) ** 2
                ).sum(-1)
            elif self.query_leaves_only:
                # Take distances between each tgt level and query leaves
                pos_distances = ((pos_path_embedded - query_path_embedded[:, -1:, :]) ** 2).sum(-1)
                neg_distances = (
                    (neg_path_embedded.unsqueeze(0) - query_path_embedded[:, -1:, :].unsqueeze(1)) ** 2
                ).sum(-1)
            elif self.tgt_leaves_only:
                # Take distances between each query level and tgt leaves
                pos_distances = ((pos_path_embedded[:, -1:, :] - query_path_embedded[:, -1:, :]) ** 2).sum(-1)
                neg_distances = (
                    (neg_path_embedded[:, -1:, :].unsqueeze(0) - query_path_embedded[:, -1:, :].unsqueeze(1)) ** 2
                ).sum(-1)
            else:
                # Take distances between each tgt level AND query level
                pos_distances = ((pos_path_embedded - query_path_embedded) ** 2).sum(-1)
                neg_distances = ((neg_path_embedded.unsqueeze(0) - query_path_embedded.unsqueeze(1)) ** 2).sum(-1)
        elif self.similarity_fn == "dot":
            pos_distances = -nn.functional.relu((pos_path_embedded * query_path_embedded).sum(-1))
            neg_distances = -nn.functional.relu(
                (neg_path_embedded.unsqueeze(0) * query_path_embedded.unsqueeze(1)).sum(-1)
            )
        elif self.similarity_fn == "cosine":
            # print(pos_path_embedded.shape, query_path_embedded.shape)
            pos_distances = -nn.functional.relu(cos_sim(pos_path_embedded, query_path_embedded))
            neg_distances = -nn.functional.relu(
                cos_sim(neg_path_embedded.unsqueeze(0), query_path_embedded.unsqueeze(1))
            )
            # print(pos_distances.shape, neg_distances.shape)
            # print(pos_path_embedded.norm(dim=-1).min(), query_path_embedded.norm(dim=-1).min(), neg_path_embedded.norm(dim=-1).min())
            # print(pos_distances.isnan().any(), neg_distances.isnan().any())
            # exit()
        else:
            raise Exception("Unrecognised similarity_fn: {:}".format(self.similarity_fn))

        pos_distances = torch.exp(
            (-pos_distances * hierarchy_weight * mask / self.tau).mean(-1)
            if self.mean_weighting
            else (-pos_distances * hierarchy_weight * mask / self.tau).sum(-1)
        )  # bsz
        neg_distances = torch.exp(
            (
                -neg_distances
                * hierarchy_weight.unsqueeze(self.hierarchy_weight_dim)
                * mask.unsqueeze(self.hierarchy_weight_dim)
                / self.tau
            ).mean(-1)
            if self.mean_weighting
            else (
                -neg_distances
                * hierarchy_weight.unsqueeze(self.hierarchy_weight_dim)
                * mask.unsqueeze(self.hierarchy_weight_dim)
                / self.tau
            ).sum(-1)
        )  # bsz x bsz

        if self.inbatch_negatives:
            # Assume scores for in-batch negs are 0 - then they only need to be included in the denominator
            numerator = pos_distances * (pos_scores if self.maximise_expected_scores else 1) + neg_distances.diagonal(
                dim1=0, dim2=1
            ) * (neg_scores if self.maximise_expected_scores else 0)

            if self.inbatch_weight is not None:
                decay_factor = (
                    exp(-float(global_step) / self.inbatch_weight_decaysteps)
                    if self.inbatch_weight_decaysteps is not None
                    else 1.0
                )

                inbatch_weight = max(
                    self.inbatch_weight * decay_factor,
                    self.inbatch_weight_min if self.inbatch_weight_min is not None else 0,
                )

                denom = (
                    pos_distances
                    + neg_distances.diagonal(dim1=0, dim2=1) * self.hard_neg_weight
                    + get_off_diagonal_elements(neg_distances * query_neg_tfidf_mask, 0, 1).mean(dim=1)
                    * inbatch_weight
                )

            else:
                denom = pos_distances + neg_distances.mean(dim=1) if self.softnn_agg_mean else neg_distances.sum(dim=1)

        else:
            print("Loss without inbatch negs has not been kept up to date! Use with caution")
            numerator = pos_distances * pos_scores + neg_distances.diag() * neg_scores
            denom = pos_distances + neg_distances.diag()

        loss = (
            -1.0 * (torch.log(numerator) - torch.log(denom)) * (pos_scores if not self.maximise_expected_scores else 1)
        )

        return loss
