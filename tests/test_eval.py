import json
import os
from datetime import datetime

import torch
from absl import app


from . import utils as test_utils

from torchseq.utils import metrics

from torchseq.utils.fleiss import fleiss

import numpy as np


def test_bleu():

    refs = ["The dog bit the man.", "It was not unexpected.", "The man bit him first."]
    sys = ["The dog bit the man.", "It wasn't surprising.", "The man had just bitten him."]
    bleu = metrics.bleu_corpus(sys, refs)
    assert abs(bleu - 45.0675) < 0.001, "BLEU score for basic examples differs from SacreBLEU reference!"


def test_f1():

    assert metrics.f1("same", "same") == 1.0, "F1 failed for correct example!"
    assert metrics.f1("same", "diff") == 0.0, "F1 failed for wrong example!"
    assert metrics.f1("tok1 tok2", "tok1 tok3") == 0.5, "F1 failed for overlapping example!"


def test_fleiss():

    bad = [[1, 1], [1, 1], [1, 1]]

    good = [[2, 0], [0, 2], [2, 0]]

    random = [[2, 0], [0, 2], [1, 1], [1, 1]]

    assert fleiss(np.array(bad)) == -1.0
    assert fleiss(np.array(good)) == 1
    assert fleiss(np.array(random)) == 0
