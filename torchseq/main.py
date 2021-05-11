#!/usr/bin/python3

import json
import os
import logging
from datetime import datetime

import torch

from torchseq.agents.aq_agent import AQAgent
from torchseq.agents.para_agent import ParaphraseAgent
from torchseq.agents.prepro_agent import PreprocessorAgent
from torchseq.agents.lm_agent import LangModelAgent
from torchseq.args import parse_args
from torchseq.utils.config import Config, merge_cfg_dicts

from torchseq.datasets.builder import dataloader_from_config

import transformers

AGENT_TYPES = {"aq": AQAgent, "langmodel": LangModelAgent, "para": ParaphraseAgent, "autoencoder": ParaphraseAgent}

"""
Entry point for torchseq CLI
"""


def main():

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s", datefmt="%H:%M"
    )
    logger = logging.getLogger("CLI")

    if args.version:
        print("Torchseq: v0.0.1")
        return

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        transformers.logging.set_verbosity_debug()

    if args.load is not None:
        args.config = os.path.join(args.load, "config.json")

        if os.path.exists(os.path.join(args.load, "model", "checkpoint.pt")):
            args.load_chkpt = os.path.join(args.load, "model", "checkpoint.pt")
        elif os.path.exists(os.path.join(args.load, "orig_model.txt")):
            with open(os.path.join(args.load, "orig_model.txt")) as f:
                chkpt_pth = f.readlines()[0]
            args.load_chkpt = chkpt_pth
        else:
            raise Exception("Tried to load from a path that has no checkpoint or pointer file: {:}".format(args.load))

    logger.info("** Running with config={:} **".format(args.config))

    with open(args.config) as f:
        cfg_dict = json.load(f)
        if args.data_path is not None:
            cfg_dict["env"]["data_path"] = args.data_path

    if args.patch is not None and len(args.patch) > 0:
        for mask_path in args.patch:
            with open(mask_path) as f:
                cfg_mask = json.load(f)
            cfg_dict = merge_cfg_dicts(cfg_dict, cfg_mask)

    if args.validate_train:
        cfg_dict["training"]["shuffle_data"] = False

    config = Config(cfg_dict)

    # Tokenizer(config.prepro.tokenizer)

    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + config.name
        + ("_TEST" if args.test else "")
        + ("_DEV" if (not args.train and args.validate) else "")
        + ("_EVALTRAIN" if args.validate_train else "")
    )

    logger.info("** Run ID is {:} **".format(run_id))

    data_loader = dataloader_from_config(config)

    if args.preprocess:
        raise Exception("Preprocessing is not currently maintained :(")
        preprocessor = PreprocessorAgent(config)
        preprocessor.logger.info("Preprocessing...")
        preprocessor.run()
        preprocessor.logger.info("...done!")
        return

    agent = AGENT_TYPES[config.task](
        config,
        run_id,
        args.output_path,
        silent=args.silent,
        verbose=(not args.silent),
        training_mode=args.train,
        profile=args.profile,
    )

    if args.load_chkpt is not None:
        logger.info("Loading from checkpoint...")
        agent.load_checkpoint(args.load_chkpt)
        logger.info("...loaded!")

    if args.train:
        logger.info("Starting training...")
        agent.train(data_loader)
        logger.info("...training done!")

    if args.reload_after_train:
        logger.info("Training done - reloading saved model")
        save_path = os.path.join(agent.output_path, agent.config.tag, agent.run_id, "model", "checkpoint.pt")
        agent.load_checkpoint(save_path, write_pointer=False)
        logger.info("...loaded!")

    if args.validate_train:
        logger.info("Starting validation (on training set)...")
        _ = agent.validate(
            data_loader, save=False, force_save_output=True, use_train=True, save_model=False, slow_metrics=True
        )
        logger.info("...validation done!")

    if args.validate:
        logger.info("Starting validation...")
        _ = agent.validate(data_loader, save=False, force_save_output=True, save_model=False, slow_metrics=True)
        logger.info("...validation done!")

    if args.test:
        logger.info("Starting testing...")
        _ = agent.validate(
            data_loader, save=False, force_save_output=True, use_test=True, save_model=False, slow_metrics=True
        )
        logger.info("...testing done!")


if __name__ == "__main__":
    main()
