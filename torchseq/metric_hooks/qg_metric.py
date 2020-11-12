from torchseq.metric_hooks.base import MetricHook
from torchseq.utils.functions import top_k_top_p_filtering, onehot
from torchseq.utils.tokenizer import Tokenizer
import torch


class QGMetricHook(MetricHook):

    type = "slow"

    def __init__(self, config, src_field=None, tgt_field=None):
        super().__init__(config, src_field, tgt_field)

    def on_begin_epoch(self):
        self.scores = {"qg_metric": []}

    def on_batch(self, batch, logits, output, memory):

        # Calc QG metric
        # Calculate metric from "On the Importance of Diversity in Question Generation for QA"
        omega = 0.7
        if self.config.get("nucleus_sampling", None) is not None:
            top_p = self.config.nucleus_sampling.cutoff
        else:
            top_p = 0.9

        nucleus_prob = torch.softmax(top_k_top_p_filtering(logits, top_p=top_p), dim=-1)
        gt_onehot = onehot(batch[self.tgt_field], N=self.config.prepro.vocab_size, ignore_index=Tokenizer().pad_id)
        accuracy = torch.sum(torch.sum(nucleus_prob * gt_onehot, dim=-1), dim=-1) / (
            batch[self.tgt_field + "_len"] - 1
        )

        diversity = torch.sum(torch.sum(torch.gt(nucleus_prob * gt_onehot, 0) * 1.0, dim=-1), dim=-1) / (
            batch[self.tgt_field + "_len"] - 1
        )

        dev_qg_metric = omega * accuracy + (1 - omega) * diversity

        self.scores["qg_metric"].extend(dev_qg_metric.tolist())
