#!/usr/bin/python3

import json
import os
from datetime import datetime

import torch
from absl import app

from agents.aq_agent import AQAgent
from agents.para_agent import ParaphraseAgent
from agents.prepro_agent import PreprocessorAgent
from args import FLAGS as FLAGS
from datasets import cqa_triple, loaders
from utils.config import Config
from utils.seed import set_seed
from utils.tokenizer import BPE


def main(_):

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

    print("** Run ID is {:} **".format(run_id))

    if FLAGS.preprocess:
        preprocessor = PreprocessorAgent(config)
        preprocessor.logger.info("Preprocessing...")
        preprocessor.run()
        preprocessor.logger.info("...done!")
        return

    if config.task == "aq":
        agent = AQAgent(config, run_id, FLAGS.output_path, silent=FLAGS.silent)
    elif config.task in ["para", "autoencoder"]:
        agent = ParaphraseAgent(config, run_id, FLAGS.output_path, silent=FLAGS.silent)

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
        agent.validate(save=False, force_save_output=True, use_train=True)
        agent.logger.info("...validation done!")

    if FLAGS.validate:
        agent.logger.info("Starting validation...")
        agent.validate(save=False, force_save_output=True)
        agent.logger.info("...validation done!")

    if FLAGS.test:
        agent.logger.info("Starting testing...")
        agent.validate(save=False, force_save_output=True, use_test=True)
        agent.logger.info("...testing done!")


if __name__ == "__main__":
    app.run(main)
