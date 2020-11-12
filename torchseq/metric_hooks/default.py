from torchseq.metric_hooks.base import MetricHook
from torchseq.utils.perplexity import get_perplexity
from torchseq.utils.tokenizer import Tokenizer
import torch


class DefaultMetricHook(MetricHook):

    type = "live"

    def __init__(self, config, src_field=None, tgt_field=None):
        super().__init__(config, src_field, tgt_field)

    def on_begin_epoch(self):
        self.scores = {"ppl": [], "nll_ext": []}

    def on_batch(self, batch, logits, output, memory):

        self.scores["ppl"].extend(
            get_perplexity(
                logits,
                batch[self.tgt_field],
                vocab_size=self.config.prepro.vocab_size,
                ignore_index=Tokenizer().pad_id,
            )
        )

        # TODO: actually calculate this, and compare to the loss that comes from the teacher forced decoder!
        self.scores["nll_ext"].extend([0] * len(batch[self.tgt_field]))
