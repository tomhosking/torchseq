import json
import os
from datetime import datetime

import torch
from absl import app


from tests import utils as test_utils

from pretrained.qa import PreTrainedQA
from pretrained.lm import PretrainedLM


@test_utils.slow
def test_qa():

    use_cuda = torch.cuda.is_available()

    instance = PreTrainedQA(device=("cuda" if use_cuda else "cpu"))

    preds = instance.infer_batch(
        ["Who was the oldest cat?", "Who was a nice puppet?"],
        ["Creme Puff was the oldest cat.", "This is a distraction. " * 50 + "Creme Puff was the oldest cat."],
    )

    assert len(preds) == 2, "Failed to produce the right number of predictions?!"
    assert preds[0] == "Creme Puff", "Short QA test failed - answer is wrong"
    assert preds[1] == "Creme Puff", "Long QA test failed - answer is wrong"


@test_utils.slow
def test_lm():

    instance = PretrainedLM()

    preds = instance.get_log_prob(
        [
            "The first thing",
            "Creme Puff was the oldest cat.",
            " Variational Autoencoders (VAEs) provide a theoretically-backed and popular framework for deep generative models.",
        ],
    )

    print(preds)

    assert len(preds) == 3, "Failed to produce the right number of predictions?!"
    assert preds[0] < 9, "Simple sentence score is too high"
    assert preds[0] > 8, "Simple sentence score is too low"
    assert abs(preds[1] - 8.8) < 0.1, "Normal sentence score out of range"
    assert abs(preds[2] - 10.8) < 0.1, "Hard sentence score out of range"
