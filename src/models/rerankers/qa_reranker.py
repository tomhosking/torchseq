import torch
import torch.nn as nn

from pretrained.qa import PreTrainedQA
from utils.metrics import f1
from utils.tokenizer import BPE


class QaReranker(nn.Module):
    def __init__(self, config, device):
        super(QaReranker, self).__init__()
        self.config = config
        self.device = device

        self.qa_model = PreTrainedQA(device=self.device)

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, presorted=True, top1=True):

        # First, stringify
        output_strings = [
            [BPE.decode(candidates.data[i][j][: lengths[i][j]]) for j in range(len(lengths[i]))]
            for i in range(len(lengths))
        ]

        qa_scores = []
        for ix, q_batch in enumerate(output_strings):
            contexts_cropped = [BPE.decode(batch["c"][ix][: batch["c_len"][ix]]) for _ in range(len(q_batch))]
            answers = self.qa_model.infer_batch(question_list=q_batch, context_list=contexts_cropped)

            this_scores = [
                scores[ix][jx] + (0 if f1(batch["a_text"][ix], ans) > 0.75 else -100) for jx, ans in enumerate(answers)
            ]

            qa_scores.append(this_scores)

        qa_scores = torch.FloatTensor(qa_scores).to(scores)
        scores = qa_scores

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
