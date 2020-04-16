import json
import os
from datetime import datetime

import torch
from absl import app


from tests import utils as test_utils

from agents.aq_agent import AQAgent
from agents.para_agent import ParaphraseAgent
from datasets import cqa_triple, loaders
from utils.config import Config
from utils.seed import set_seed
from utils.tokenizer import BPE


@test_utils.slow
def test_bert_embeds():

    CONFIG = "./models/optimised/bert_embeds/20200113_075322_0sent_lr3e-3/config.json"
    CHKPT = "./models/optimised/bert_embeds/20200113_075322_0sent_lr3e-3/model/checkpoint.pth.tar"
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./runs/"
    SEED = 123

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    assert use_cuda, "This test needs to run on GPU!"

    with open(CONFIG) as f:
        cfg_dict = json.load(f)
        if DATA_PATH is not None:
            cfg_dict["env"]["data_path"] = DATA_PATH

        config = Config(cfg_dict)

    set_seed(SEED)

    # This is not a good way of passing this value in
    BPE.pad_id = config.prepro.vocab_size
    BPE.embedding_dim = config.embedding_dim
    BPE.model_slug = config.encdec.bert_model

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.name + "_REGRESSION"

    if config.task == "aq":
        agent = AQAgent(config, run_id, OUTPUT_PATH, silent=True)
    elif config.task in ["para", "autoencoder"]:
        agent = ParaphraseAgent(config, run_id, OUTPUT_PATH, silent=True)

    agent.load_checkpoint(CHKPT)
    loss, metrics = agent.validate(save=False, force_save_output=True)

    # Now check the output
    assert abs(loss.item() - 2.5841) < 1e-3, "Loss is different to expected!"  # 2.3993
    assert "bleu" in metrics, "BLEU is missing from output metrics!"
    assert abs(metrics["bleu"] - 17.989) < 1e-2, "BLEU score is different to expected!"
