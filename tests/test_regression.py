import json
import os
from datetime import datetime

import torch
from absl import app


from . import utils as test_utils

from torchseq.agents.aq_agent import AQAgent
from torchseq.agents.para_agent import ParaphraseAgent
from torchseq.datasets import qa_triple, loaders
from torchseq.utils.config import Config
from torchseq.utils.seed import set_seed
from torchseq.utils.tokenizer import Tokenizer


@test_utils.slow
def test_bert_embeds():

    CONFIG = "./models/optimised/aq/20200729_141718_optimised_embeds/config.json"
    CHKPT = "./models/optimised/aq/20200729_141718_optimised_embeds/model/checkpoint.pth.tar"
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./runs/"
    SEED = 123

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()

    assert use_cuda, "This test needs to run on GPU!"

    with open(CONFIG) as f:
        cfg_dict = json.load(f)
        if DATA_PATH is not None:
            cfg_dict["env"]["data_path"] = DATA_PATH

        cfg_dict["eval"]["truncate_dataset"] = 100

        config = Config(cfg_dict)

    set_seed(SEED)

    # This is not a good way of passing this value in
    # Tokenizer(config.prepro.tokenizer).reload(config.prepro.tokenizer)

    # run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.name + "_REGRESSION"

    if config.task == "aq":
        agent = AQAgent(config, None, OUTPUT_PATH, silent=True)
    elif config.task in ["para", "autoencoder"]:
        agent = ParaphraseAgent(config, None, OUTPUT_PATH, silent=True)

    agent.load_checkpoint(CHKPT)
    loss, metrics, memory = agent.validate(save=False, force_save_output=True, save_model=False)

    # Now check the output (for first 100 samples)
    assert abs(loss.item() - 3.247) < 1e-3, "Loss is different to expected!"
    assert "bleu" in metrics, "BLEU is missing from output metrics!"
    assert abs(metrics["bleu"] - 15.35) < 1e-2, "BLEU score is different to expected!"

    # Targets for full dataset:
    # assert abs(loss.item() - 2.833) < 1e-3, "Loss is different to expected!"
    # assert "bleu" in metrics, "BLEU is missing from output metrics!"
    # assert abs(metrics["bleu"] - 17.22) < 1e-2, "BLEU score is different to expected!"


@test_utils.slow
def test_autoencoder():

    CONFIG = "./models/optimised/ae/20200728_195510_squad_baseline/config.json"
    CHKPT = "./models/optimised/ae/20200728_195510_squad_baseline/model/checkpoint.pth.tar"
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./runs/"
    SEED = 123

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()

    assert use_cuda, "This test needs to run on GPU!"

    with open(CONFIG) as f:
        cfg_dict = json.load(f)
        if DATA_PATH is not None:
            cfg_dict["env"]["data_path"] = DATA_PATH

        cfg_dict["eval"]["truncate_dataset"] = 100

        config = Config(cfg_dict)

    set_seed(SEED)

    # This is not a good way of passing this value in
    # Tokenizer(config.prepro.tokenizer).reload(config.prepro.tokenizer)

    # run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.name + "_REGRESSION"

    if config.task == "aq":
        agent = AQAgent(config, None, OUTPUT_PATH, silent=True)
    elif config.task in ["para", "autoencoder"]:
        agent = ParaphraseAgent(config, None, OUTPUT_PATH, silent=True)

    agent.load_checkpoint(CHKPT)
    loss, metrics, memory = agent.validate(save=False, force_save_output=True, save_model=False)

    # Now check the output (for first 100 samples)
    assert abs(loss.item() - 0.0974) < 1e-3, "Loss is different to expected!"
    assert "bleu" in metrics, "BLEU is missing from output metrics!"
    assert abs(metrics["bleu"] - 92.13) < 1e-2, "BLEU score is different to expected!"

    # Targets for full dataset:
    # assert abs(loss.item() - ???) < 1e-3, "Loss is different to expected!"
    # assert "bleu" in metrics, "BLEU is missing from output metrics!"
    # assert abs(metrics["bleu"] - ???) < 1e-2, "BLEU score is different to expected!"
