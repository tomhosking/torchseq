class MetricHook:

    type: str  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation

    def __init__(self, config, tokenizer, src_field=None, tgt_field=None):
        self.config = config
        self.tokenizer = tokenizer
        self.src_field = src_field
        self.tgt_field = tgt_field

    def on_begin_epoch(self, use_test=False):
        self.scores = {}

    def on_batch(self, batch, logits, output, memory, use_test=False):
        raise NotImplementedError("You need to implement on_batch for your MetricHook!")

    def on_end_epoch(self, agent, use_test=False):
        return {k: sum(v) / len(v) for k, v in self.scores.items()}
