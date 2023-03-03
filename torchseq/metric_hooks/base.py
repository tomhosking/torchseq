from typing import Dict, List, Optional, Any

import torch
from torchseq.utils.config import Config
from torchseq.utils.tokenizer import Tokenizer
from torchseq.agents.base import BaseAgent


class MetricHook:
    type: str  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation
    config: Config
    tokenizer: Tokenizer
    src_field: Optional[str]
    tgt_field: Optional[str]
    scores: Dict[str, List[float]]

    def __init__(
        self, config: Config, tokenizer: Tokenizer, src_field: Optional[str] = None, tgt_field: Optional[str] = None
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.src_field = src_field
        self.tgt_field = tgt_field

    def on_begin_epoch(self, use_test: bool = False):
        self.scores = {}

    def on_batch(
        self,
        batch: Dict[str, torch.Tensor],
        logits: torch.Tensor,
        output: List[str],
        memory: Dict[str, torch.Tensor],
        use_test: bool = False,
    ):
        raise NotImplementedError("You need to implement on_batch for your MetricHook!")

    def on_end_epoch(self, agent: BaseAgent, use_test: bool = False) -> Dict[str, float]:
        return {k: sum(v) / len(v) for k, v in self.scores.items()}
