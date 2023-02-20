import json
import os
from datetime import datetime

import torch


from . import utils as test_utils

from torchseq.utils import metrics
from torchseq.utils.rouge import get_jackknife_rouge, get_pairwise_rouge

from torchseq.utils.fleiss import fleiss

import numpy as np


def test_bleu():

    refs = ["The dog bit the man.", "It was not unexpected.", "The man bit him first."]
    sys = ["The dog bit the man.", "It wasn't surprising.", "The man had just bitten him."]
    bleu = metrics.bleu_corpus(sys, refs)
    assert abs(bleu - 45.0675) < 0.001, "BLEU score for basic examples differs from SacreBLEU reference!"


def test_rouge():

    refs = ["The dog bit the man.", "It was not unexpected.", "The man bit him first."]
    sys = ["The dog bit the man.", "It wasn't surprising.", "The man had just bitten him."]
    rouge = get_pairwise_rouge(sys[0], refs[0])
    assert 'rouge2' in rouge
    assert 'rougeL' in rouge
    assert abs(rouge['rouge2'] - 100) < 0.001, "Rouge score for basic examples differs from reference!"

    refs = [["The dog bit the man.", "The man was bitten by the dog"], ["It was not unexpected.", "it was not surprising"], ["The man bit him first."]]
    rouge = get_jackknife_rouge(sys, refs)
    assert 'rouge2' in rouge
    assert 'rougeL' in rouge
    assert abs(rouge['rouge2'] - 30.741) < 0.001, "Rouge score for jackknife examples differs from reference!"

    


def test_meteor():
    preds = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    refs = ["It is a guide to action that ensures that the military will forever heed Party commands"]

    assert metrics.meteor_corpus(refs, preds) - 0.7398 < 1e-4


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
