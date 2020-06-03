import json
import os
from datetime import datetime

import torch
from absl import app


from tests import utils as test_utils

from pretrained.qa import PreTrainedQA


@test_utils.slow
def test_qa():

    use_cuda = torch.cuda.is_available()

    instance = PreTrainedQA(device="cpu")

    preds = instance.infer_batch(
        ["Who was the oldest cat?", "Who was a nice puppet?"],
        ["Creme Puff was the oldest cat.", "This is a distraction. " * 50 + "Creme Puff was the oldest cat."],
    )

    assert len(preds) == 2, "Failed to produce the right number of predictions?!"
    assert preds[0] == "Creme Puff", "Short QA test failed - answer is wrong"
    assert preds[1] == "Creme Puff", "Long QA test failed - answer is wrong"
