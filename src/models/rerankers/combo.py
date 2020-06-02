import torch
import torch.nn as nn

from utils.tokenizer import BPE

from models.rerankers.qa_reranker import QaReranker
from models.rerankers.topk import TopkReducer
from models.rerankers.ngram_reranker import NgramReranker
from models.rerankers.backtranslate_reranker import BacktranslateReranker


class CombinationReranker(nn.Module):
    def __init__(self, config, device, src_field, model):
        super(CombinationReranker, self).__init__()
        self.config = config
        self.device = device

        self.src_field = src_field
        self.model = model

        self.qa_reranker = QaReranker(self.config, self.device)
        self.ngram_reranker = NgramReranker(self.config, self.device, self.src_field)
        self.backtranslate_reranker = BacktranslateReranker(self.config, self.device, self.src_field, self.model)

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, sort=True, top1=True):

        # store the original seq probs
        nll_scores = scores

        # ngram scores are fraction of overlapping toks (lower is better)
        _, _, ngram_scores = self.ngram_reranker(candidates, lengths, batch, tgt_field, top1=False, sort=False)

        # backtrans scores are nll of recovering original from candidate (lower is better)
        _, _, backtrans_scores = self.backtranslate_reranker(
            candidates, lengths, batch, tgt_field, top1=False, sort=False
        )

        # qa scores are F1 score
        _, _, qa_scores = self.qa_reranker(candidates, lengths, batch, tgt_field, top1=False, sort=False)

        # QA score should dominate - but if all candidates are unanswerable, then fall back on other scores
        scores = (ngram_scores * 1.5 + (backtrans_scores + nll_scores) / 2) * (qa_scores * 0.9 + 0.1)

        if sort:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)

            sorted_seqs = torch.gather(candidates, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, candidates.shape[2]))

        if top1:
            output = sorted_seqs[:, 0, :]
        else:
            topk = self.config.eval.data.get("topk", None)
            if topk is not None:
                output = sorted_seqs[:, :topk, :]
            else:
                output = sorted_seqs[:, 0, :]

        return output, torch.sum(output != BPE.pad_id, dim=-1), sorted_scores
