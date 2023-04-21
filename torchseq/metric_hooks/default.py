from torchseq.metric_hooks.base import MetricHook
from torchseq.utils.perplexity import get_perplexity
from torchseq.utils.tokenizer import Tokenizer
import torch


class DefaultMetricHook(MetricHook):
    type = "live"

    # def __init__(self, config, tokenizer, src_field=None, tgt_field=None):
    #     super().__init__(config, tokenizer, src_field, tgt_field)

    def on_begin_epoch(self, use_test=False):
        self.scores = {"ppl": []}

    def on_batch(self, batch, logits, output, memory, use_test=False):
        if self.tgt_field is not None:
            self.scores["ppl"].extend(
                get_perplexity(
                    logits,
                    batch[self.tgt_field],
                    vocab_size=self.config.prepro.get_first(["output_vocab_size", "vocab_size"]),
                    ignore_index=self.tokenizer.pad_id,
                ).tolist()
            )
