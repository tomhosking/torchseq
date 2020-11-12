class MetricHook:

    type = None  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation

    def __init__(self, config, src_field=None, tgt_field=None):
        self.config = config
        self.src_field = src_field
        self.tgt_field = tgt_field

    def on_begin_epoch(self):
        self.scores = {}

    def on_batch(self, batch, logits, output, memory):
        raise NotImplementedError("You need to implement on_batch for your MetricHook!")

    def on_end_epoch(self):
        return {k: sum(v) / len(v) for k, v in self.scores.items()}
