from tests import utils as test_utils


from utils.tokenizer import BPE
import torch


def test_bert_uncased_tokenizer():
    BPE.pad_id = 30522
    BPE.embedding_dim = 512
    BPE.model_slug = 'bert-base-uncased'

    TEST_STRING = 'This is a test sentence.'

    BPE.reload()

    tokenized = BPE.tokenise(TEST_STRING)
    decoded = BPE.decode(torch.LongTensor([tok['id'] for tok in tokenized]))

    assert [tok['id'] for tok in tokenized] == [101, 2023, 2003, 1037, 3231, 6251, 1012, 102]
    assert decoded == TEST_STRING.lower()


def test_bert_cased_tokenizer():
    BPE.pad_id = 28996
    BPE.embedding_dim = 512
    BPE.model_slug = 'bert-base-cased'

    TEST_STRING = 'This is a test sentence.'

    BPE.reload()

    tokenized = BPE.tokenise(TEST_STRING)
    decoded = BPE.decode(torch.LongTensor([tok['id'] for tok in tokenized]))

    assert [tok['id'] for tok in tokenized] == [101, 1188, 1110, 170, 2774, 5650, 119, 102]
    assert decoded == TEST_STRING

