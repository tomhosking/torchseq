#!/usr/bin/python3

import json
import os
from datetime import datetime

import torch
from absl import app

from torchseq.agents.aq_agent import AQAgent
from torchseq.agents.para_agent import ParaphraseAgent
from torchseq.agents.prepro_agent import PreprocessorAgent
from torchseq.agents.lm_agent import LangModelAgent
from torchseq.args import FLAGS as FLAGS
from torchseq.datasets import qa_triple, loaders
from torchseq.utils.config import Config, merge_cfg_dicts
from torchseq.utils.seed import set_seed
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.logging import Logger


def main(_):

    if FLAGS.load is not None:
        FLAGS.config = os.path.join(FLAGS.load, "config.json")

        if os.path.exists(os.path.join(FLAGS.load, "orig_model.txt")):
            with open(os.path.join(FLAGS.load, "orig_model.txt")) as f:
                chkpt_pth = f.readlines()[0]
            FLAGS.load_chkpt = chkpt_pth
        else:
            FLAGS.load_chkpt = os.path.join(FLAGS.load, "model", "checkpoint.pt")

    print("** Running with config={:} **".format(FLAGS.config))

    with open(FLAGS.config) as f:
        cfg_dict = json.load(f)
        if FLAGS.data_path is not None:
            cfg_dict["env"]["data_path"] = FLAGS.data_path

    if FLAGS.patch is not None and len(FLAGS.patch) > 0:
        for mask_path in FLAGS.patch:
            with open(mask_path) as f:
                cfg_mask = json.load(f)
            cfg_dict = merge_cfg_dicts(cfg_dict, cfg_mask)

    if FLAGS.validate_train:
        cfg_dict["training"]["shuffle_data"] = False

    config = Config(cfg_dict)

    Tokenizer(config.prepro.tokenizer)

    set_seed(FLAGS.seed)

    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + config.name
        + ("_TEST" if FLAGS.test else "")
        + ("_DEV" if FLAGS.validate else "")
        + ("_EVALTRAIN" if FLAGS.validate_train else "")
    )

    print("** Run ID is {:} **".format(run_id))

    if FLAGS.preprocess:
        preprocessor = PreprocessorAgent(config)
        preprocessor.logger.info("Preprocessing...")
        preprocessor.run()
        preprocessor.logger.info("...done!")
        return

    agents = {"aq": AQAgent, "langmodel": LangModelAgent, "para": ParaphraseAgent, "autoencoder": ParaphraseAgent}

    agent = agents[config.task](config, run_id, FLAGS.output_path, silent=FLAGS.silent, training_mode=FLAGS.train)

    if FLAGS.load_chkpt is not None:
        agent.logger.info("Loading from checkpoint...")
        agent.load_checkpoint(FLAGS.load_chkpt)
        agent.logger.info("...loaded!")

    if FLAGS.train:
        agent.logger.info("Starting training...")
        agent.train()
        agent.logger.info("...training done!")

    if FLAGS.validate_train:
        agent.logger.info("Starting validation (on training set)...")
        agent.validate(save=False, force_save_output=True, use_train=True, save_model=False)
        agent.logger.info("...validation done!")

    if FLAGS.validate:
        agent.logger.info("Starting validation...")
        agent.validate(save=False, force_save_output=True, save_model=False)
        agent.logger.info("...validation done!")

    if FLAGS.test:
        agent.logger.info("Starting testing...")
        agent.validate(save=False, force_save_output=True, use_test=True, save_model=False)
        agent.logger.info("...testing done!")


def run():
    app.run(main)
