import torch

from tests import utils as test_utils
from utils.tokenizer import BPE


def test_bert_uncased_basic():
    BPE.pad_id = 30522
    BPE.embedding_dim = 512
    BPE.model_slug = "bert-base-uncased"

    TEST_STRING = "This is a test sentence."

    BPE.reload()

    tokenized = BPE.tokenise(TEST_STRING)
    decoded = BPE.decode(torch.LongTensor([tok["id"] for tok in tokenized]))

    assert [tok["id"] for tok in tokenized] == [101, 2023, 2003, 1037, 3231, 6251, 1012, 102], "BERT uncased tok ids are wrong for basic example!"
    assert decoded == TEST_STRING.lower(),  "BERT uncased tokenisation isn't reversible for basic example!"


def test_bert_cased_basic():
    BPE.pad_id = 28996
    BPE.embedding_dim = 512
    BPE.model_slug = "bert-base-cased"

    TEST_STRING = "This is a test sentence."

    BPE.reload()

    tokenized = BPE.tokenise(TEST_STRING)
    decoded = BPE.decode(torch.LongTensor([tok["id"] for tok in tokenized]))

    assert [tok["id"] for tok in tokenized] == [101, 1188, 1110, 170, 2774, 5650, 119, 102], "BERT cased tok ids are wrong for basic example!"
    assert decoded == TEST_STRING,  "BERT cased tokenisation isn't reversible for basic example!"
