from tests import utils as test_utils


import json
import os
from datetime import datetime

import torch
from absl import app

from agents.aq_agent import AQAgent
from agents.para_agent import ParaphraseAgent
from args import FLAGS as FLAGS
from datasets import cqa_triple, loaders
from utils.config import Config
from utils.seed import set_seed
from utils.tokenizer import BPE

@test_utils.slow
def test_bert_embeds():

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    print("** Running with config={:} **".format(FLAGS.config))

    with open(FLAGS.config) as f:
        cfg_dict = json.load(f)
        if FLAGS.data_path is not None:
            cfg_dict["env"]["data_path"] = FLAGS.data_path

        config = Config(cfg_dict)

    set_seed(FLAGS.seed)

    # This is not a good way of passing this value in
    BPE.pad_id = config.prepro.vocab_size
    BPE.embedding_dim = config.embedding_dim
    BPE.model_slug = config.encdec.bert_model

    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + config.name
        + ("_TEST" if FLAGS.test else "")
        + ("_DEV" if FLAGS.validate else "")
        + ("_EVALTRAIN" if FLAGS.validate_train else "")
    )

    if config.task == "aq":
        agent = AQAgent(config, run_id, silent=FLAGS.silent)
    elif config.task in ["para", "autoencoder"]:
        agent = ParaphraseAgent(config, run_id, silent=FLAGS.silent)

    agent.load_checkpoint(FLAGS.load_chkpt)
    loss, metrics = agent.validate(save=False, force_save_output=True)

    # Now check the output
    assert abs(loss - 2.3993) < 1e-3
    assert 'bleu' in metrics
    assert abs(metrics['bleu'] - 17.989) < 1e-2
