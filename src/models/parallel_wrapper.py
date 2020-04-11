import torch
import torch.nn as nn


class ParallelModel(nn.module):
    def __init__(self, model, loss, tgt_field):
        super(ParallelModel, self).__init__()

        self.model = model
        self.loss = loss

        self.tgt_field = tgt_field

    def forward(self, batch, *args, **kwargs):

        res = self.model(batch, *args, **kwargs)

        this_loss = self.loss(res[0].permute(0, 2, 1), batch[self.tgt_field])

        loss = torch.mean(torch.sum(this_loss, dim=1) / batch[self.tgt_field + "_len"].to(this_loss), dim=0)

        ret = (loss, *res)

        return ret


def parallelify(model, loss, tgt_field):
    return nn.DataParallel(ParallelModel(model, loss, tgt_field))
