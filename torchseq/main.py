#!/usr/bin/python3

import json
import os
import logging
from datetime import datetime

import torch
import torchseq
from torchseq.utils.wandb import wandb_log

from torchseq.agents.aq_agent import AQAgent
from torchseq.agents.seq2seq_agent import Seq2SeqAgent
from torchseq.agents.lm_agent import LangModelAgent
from torchseq.agents.meta_learning_agent import MetaLearningAgent
from torchseq.agents.exemplar_agent import ExemplarGuidedAgent
from torchseq.args import parse_args
from torchseq.utils.config import Config, merge_cfg_dicts
from torchseq.utils.mckenzie import set_status_mckenzie
from torchseq.utils.wandb import wandb_init

from torchseq.datasets.builder import dataloader_from_config

# from pytorch_lightning.lite import LightningLite
import transformers


AGENT_TYPES = {
    "aq": AQAgent,
    "langmodel": LangModelAgent,
    "para": Seq2SeqAgent,
    "seq2seq": Seq2SeqAgent,
    "autoencoder": Seq2SeqAgent,
    "metalearning": MetaLearningAgent,
    "exemplarguided": ExemplarGuidedAgent,
}

"""
Entry point for torchseq CLI
"""


# class TorchseqJob(LightningLite):
#     def run(self, args):
def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s", datefmt="%H:%M"
    )
    logger = logging.getLogger("CLI")

    args = parse_args()

    if args.version:
        print(torchseq.__version__)
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

    if args.patch is not None and len(args.patch) > 0:
        for mask_path in args.patch:
            logger.info(f"Applying patch: {mask_path}")
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

    wandb_init(config=config, run_id=run_id, path=os.path.join(args.output_path, config.tag, run_id))
    # wandb_log({"status": "running"}, 0)

    data_loader = dataloader_from_config(config, data_path=args.data_path)

    agent = AGENT_TYPES[config.task](
        config,
        run_id,
        args.output_path,
        data_path=args.data_path,
        silent=args.silent,
        verbose=args.verbose,
        training_mode=args.train,
        cache_root=(
            args.load_chkpt.replace("model/checkpoint.pt", "")
            if (args.load_chkpt is not None and not args.nocache)
            else None
        ),
        use_cuda=(not args.cpu),
    )

    # Setup Lightning
    # for optimizer in agent.optimizers.optimizers:
    #     agent.model, optimizer = self.setup(agent.model, optimizer)
    # agent.backward = self.backward

    # if data_loader._train.exists:
    #     data_loader.train_loader = self.setup_dataloaders(data_loader.train_loader)
    # if data_loader._valid.exists:
    #     data_loader.valid_loader = self.setup_dataloaders(data_loader.valid_loader)
    # if data_loader._test.exists:
    #     data_loader.test_loader = self.setup_dataloaders(data_loader.test_loader)

    if args.load_chkpt is not None:
        logger.info("Loading from checkpoint...")
        agent.load_checkpoint(args.load_chkpt)
        logger.info("...loaded!")

    if args.train:

        if data_loader.train_loader is None:
            raise Exception("Selected dataset does not include a training split - cannot train!")
        logger.info("Starting training...")
        # wandb_log({"status": "training"}, 0)

        # TEMP: save out the VQ embeds *before* they've been trained, for debug
        if agent.config.get_path(["bottleneck", "modular"], False):
            bneck_types = [x.type for x in agent.config.bottleneck.modules]
            quantizer_index = None
            if "vqvae" in bneck_types:
                quantizer_index = bneck_types.index("vqvae")
            if "hrqvae" in bneck_types:
                quantizer_index = bneck_types.index("hrqvae")
            if quantizer_index is not None:
                torch.save(
                    agent.model.bottleneck.module_list[quantizer_index].quantizer._embedding,
                    agent.run_output_path + "/vqembeds_pre.pt",
                )

        agent.train(data_loader)
        logger.info("...training done!")

    if args.reload_after_train:
        logger.info("Training done - reloading saved model")
        save_path = os.path.join(agent.run_output_path, "model", "checkpoint.pt")
        agent.load_checkpoint(save_path, write_pointer=False)
        logger.info("...loaded!")

    if args.validate_train:
        if data_loader.train_loader is None:
            raise Exception("Selected dataset does not include a training split - cannot train!")

        set_status_mckenzie("validating")
        # wandb_log({"status": "validating"}, step=agent.global_step)
        logger.info("Starting validation (on training set)...")
        _ = agent.validate(data_loader, force_save_output=True, use_train=True, save_model=False, slow_metrics=True)
        logger.info("...validation done!")

    if args.validate:
        if data_loader.valid_loader is None:
            raise Exception("Selected dataset does not include a dev split - cannot run validation!")

        # TEMP: save out the VQ embeds *after* they've been trained, for debug
        if agent.config.bottleneck.get("modular", False):
            bneck_types = [x.type for x in agent.config.bottleneck.modules]
            quantizer_index = None
            if "vqvae" in bneck_types:
                quantizer_index = bneck_types.index("vqvae")
            if "hrqvae" in bneck_types:
                quantizer_index = bneck_types.index("hrqvae")
            if quantizer_index is not None:
                torch.save(
                    agent.model.bottleneck.module_list[quantizer_index].quantizer._embedding,
                    agent.run_output_path + "/vqembeds_post.pt",
                )

        set_status_mckenzie("validating")
        # wandb_log({"status": "validating"}, step=agent.global_step)
        logger.info("Starting validation...")
        _ = agent.validate(data_loader, force_save_output=True, save_model=False, slow_metrics=True)
        logger.info("...validation done!")

    if args.test:
        if data_loader.test_loader is None:
            raise Exception("Selected dataset does not include a test split - cannot run test evaluation!")

        set_status_mckenzie("validating")
        # wandb_log({"status": "testing"}, step=agent.global_step)
        logger.info("Starting testing...")
        _ = agent.validate(data_loader, force_save_output=True, use_test=True, save_model=False, slow_metrics=True)
        logger.info("...testing done!")


# def main():
#     logging.basicConfig(
#         level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s", datefmt="%H:%M"
#     )
#     logger = logging.getLogger("CLI")

#     args = parse_args()
#     device = "cpu" if args.cpu else "gpu"
#     if torch.cuda.device_count() > 1:
#         logger.warn("Multiple ({:}) GPUs available:  not currently supported!!!".format(torch.cuda.device_count()))

#     # num_devices = torch.cuda.device_count()

#     job = TorchseqJob(accelerator=device, devices=1)  # , strategy="ddp"
#     job.run(args)


if __name__ == "__main__":
    main()
