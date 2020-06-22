import json
import os
from datetime import datetime

import torch
from absl import app


from . import utils as test_utils

from torchseq.utils.config import Config, merge_cfg_dicts


def test_config():

    main_cfg_dict = {"int": 1, "float": 0.2, "str": "hello", "bool": True, "nested": {"value": "present"}}

    mask_cfg_dict = {"int": 2, "nested": {"value": "overwritten", "newval": "added"}}

    cfg_obj = Config(main_cfg_dict)

    merged_obj = Config(merge_cfg_dicts(main_cfg_dict, mask_cfg_dict))

    assert cfg_obj.int == 1
    assert cfg_obj.float == 0.2
    assert cfg_obj.str == "hello"
    assert cfg_obj.bool is True
    assert cfg_obj.nested.value == "present"

    assert cfg_obj.get("str", "not hello") == "hello"
    assert cfg_obj.get("missing", "default_val") == "default_val"

    assert merged_obj.int == 2
    assert merged_obj.nested.value == "overwritten"
    assert merged_obj.nested.newval == "added"
