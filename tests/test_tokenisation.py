import torch

from . import utils as test_utils
from torchseq.utils.tokenizer import Tokenizer


def test_bert_uncased_basic():

    TEST_STRING = "This is a test sentence."

    Tokenizer("bert-base-uncased").reload("bert-base-uncased")

    tokenized = Tokenizer().tokenise(TEST_STRING)
    decoded = Tokenizer().decode(torch.LongTensor([tok["id"] for tok in tokenized]))

    assert [tok["id"] for tok in tokenized] == [
        101,
        2023,
        2003,
        1037,
        3231,
        6251,
        1012,
        102,
    ], "BERT uncased tok ids are wrong for basic example!"
    assert decoded == TEST_STRING.lower(), "BERT uncased tokenisation isn't reversible for basic example!"


def test_bert_cased_basic():

    TEST_STRING = "This is a test sentence."

    Tokenizer("bert-base-cased").reload("bert-base-cased")

    tokenized = Tokenizer().tokenise(TEST_STRING)
    decoded = Tokenizer().decode(torch.LongTensor([tok["id"] for tok in tokenized]))

    assert [tok["id"] for tok in tokenized] == [
        101,
        1188,
        1110,
        170,
        2774,
        5650,
        119,
        102,
    ], "BERT cased tok ids are wrong for basic example!"
    assert decoded == TEST_STRING, "BERT cased tokenisation isn't reversible for basic example!"


def test_roberta_basic():
    Tokenizer("roberta-base").reload("roberta-base")

    TEST_STRING = "This is a test sentence."


    tokenized = Tokenizer().tokenise(TEST_STRING)
    decoded = Tokenizer().decode(torch.LongTensor([tok["id"] for tok in tokenized]))

    assert [tok["id"] for tok in tokenized] == [
        0,
        713,
        16,
        10,
        1296,
        3645,
        4,
        2,
    ], "RoBERTa tok ids are wrong for basic example!"
    assert decoded == TEST_STRING, "RoBERTa tokenisation isn't reversible for basic example!"
