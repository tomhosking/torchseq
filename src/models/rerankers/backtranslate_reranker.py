import torch
import torch.nn as nn

from utils.tokenizer import BPE


class BacktranslateReranker(nn.Module):
    def __init__(self, config, device, src_field, model, decoder, loss):
        super(BacktranslateReranker, self).__init__()
        self.config = config
        self.device = device

        self.src_field = src_field
        self.model = model
        self.decoder = decoder
        self.loss = loss

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, presorted=True, top1=True):

        # _, logits, _ = self.decode_teacher_force(self.model, batch, tgt_field)
        # this_loss = self.loss(logits.permute(0, 2, 1), batch[tgt_field])
        # normed_loss = torch.mean(
        #     torch.sum(this_loss, dim=1) / (batch[tgt_field + "_len"] - 1).to(this_loss), dim=0
        # )

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
