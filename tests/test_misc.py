import json
import os
from datetime import datetime

import torch


from . import utils as test_utils

from torchseq.utils.config import Config, merge_cfg_dicts
from torchseq.utils.singleton import Singleton
import torchseq.utils.functions as tsfunctions


def test_config():

    main_cfg_dict = {
        "name": "model_name",
        "int": 1,
        "float": 0.2,
        "str": "hello",
        "bool": True,
        "nested": {"value": "present"},
    }

    mask_cfg_dict = {"name": "%_mod", "int": 2, "nested": {"value": "overwritten", "newval": "added"}}

    cfg_obj = Config(main_cfg_dict)

    merged_obj = Config(merge_cfg_dicts(main_cfg_dict, mask_cfg_dict))

    assert cfg_obj.int == 1
    assert cfg_obj.float == 0.2
    assert cfg_obj.str == "hello"
    assert cfg_obj.bool is True
    assert cfg_obj.nested.value == "present"

    assert cfg_obj.get_path(["nested", "value"]) == "present"
    assert cfg_obj.get_path(["nested", "missing"], "alt_default") == "alt_default"

    assert cfg_obj.get("str", "not hello") == "hello"
    assert cfg_obj.get("missing", "default_val") == "default_val"

    assert cfg_obj.get_first(["float", "int"]) == 0.2

    assert merged_obj.int == 2
    assert merged_obj.nested.value == "overwritten"
    assert merged_obj.nested.newval == "added"
    assert merged_obj.name == "model_name_mod"


def test_singleton():
    class TestClass(metaclass=Singleton):
        def __init__(self, val):
            self.val = val

    x = TestClass("x")
    y = TestClass("y")

    assert x.val == "x"
    assert x.val == y.val


def test_cache():
    from torchseq.utils.cache import Cache

    cache = Cache()

    data_str = "testing"
    cache.save("string", data_str)
    assert cache.load("string") == data_str

    data_tensor = torch.rand(4, 7)
    cache.save("tensor", data_tensor)
    assert cache.load("tensor").equal(data_tensor)

    assert cache.load("missing") is None


def test_functions():
    # topk
    input_probs = torch.Tensor([[0.1, 0.2, 0.3, 0.4], [0.13, 0.7, 0.17, 0.0]])

    assert tsfunctions.top_k_top_p_filtering(input_probs, top_k=2).equal(
        torch.tensor([[-torch.inf, -torch.inf, 0.3000, 0.4000], [-torch.inf, 0.7000, 0.1700, -torch.inf]])
    )

    assert tsfunctions.top_k_top_p_filtering(input_probs, top_k=2, filter_value=0).equal(
        torch.tensor([[0.0000, 0.0000, 0.3000, 0.4000], [0.0000, 0.7000, 0.1700, 0.0000]])
    )

    assert tsfunctions.top_k_top_p_filtering(input_probs, top_p=0.5, filter_value=0).equal(
        torch.tensor([[0.0000, 0.0000, 0.3000, 0.4000], [0.0000, 0.7000, 0.1700, 0.0000]])
    )

    x = torch.Tensor([[0.0, 0.0, 0.0, 1.0], [0.13, 0.7, 0.17, 0.0]])
    y = torch.Tensor([[0.0, 0.0, 0.0, 1.0]])

    assert tsfunctions.cos_sim(x, y).equal(torch.tensor([[1.0], [0.0000]]))

    test_data = list(range(6))

    assert list(tsfunctions.batchify(test_data, 4)) == [(0, [0, 1, 2, 3]), (1, [4, 5])]
