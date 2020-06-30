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

    CONFIG = "./models/optimised/bert_embeds/20200113_075322_0sent_lr3e-3/config.json"
    CHKPT = "./models/optimised/bert_embeds/20200113_075322_0sent_lr3e-3/model/checkpoint.pth.tar"
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

        config = Config(cfg_dict)

    set_seed(SEED)

    # This is not a good way of passing this value in
    Tokenizer(config.encdec.bert_model).reload(config.encdec.bert_model)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.name + "_REGRESSION"

    if config.task == "aq":
        agent = AQAgent(config, run_id, OUTPUT_PATH, silent=True)
    elif config.task in ["para", "autoencoder"]:
        agent = ParaphraseAgent(config, run_id, OUTPUT_PATH, silent=True)

    agent.load_checkpoint(CHKPT)
    loss, metrics = agent.validate(save=False, force_save_output=True)

    # Now check the output
    assert abs(loss.item() - 2.5665) < 1e-3, "Loss is different to expected!"
    assert "bleu" in metrics, "BLEU is missing from output metrics!"
    assert abs(metrics["bleu"] - 17.338) < 1e-2, "BLEU score is different to expected!"
