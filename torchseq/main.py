#!/usr/bin/python3

import json
import os
import logging
from datetime import datetime
import uuid

import torch
import torchseq


from torchseq.agents.aq_agent import AQAgent
from torchseq.agents.seq2seq_agent import Seq2SeqAgent
from torchseq.agents.lm_agent import LangModelAgent
from torchseq.args import parse_args
from torchseq.utils.config import Config, merge_cfg_dicts
from torchseq.utils.config_migration import check_config
from torchseq.utils.mckenzie import set_status_mckenzie
from torchseq.utils.wandb import wandb_init, wandb_log
from torchseq.utils.model_loader import AGENT_TYPES

from torchseq.datasets.builder import dataloader_from_config

import lightning

import transformers


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

    if args.amp:
        logger.info("Using matmul precision = high")
        torch.set_float32_matmul_precision("high")

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

    if check_config(cfg_dict):
        logger.warning("Config is outdated! Run the migration script to update it")

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
        + str(uuid.uuid4())[:4]
        + "_"
        + config.name
        + ("_TEST" if args.test else "")
        + ("_DEV" if (not args.train and args.validate) else "")
        + ("_EVALTRAIN" if args.validate_train else "")
    )

    logger.info("** Run ID is {:} **".format(run_id))

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

    wandb_init(config=config, run_id=run_id, path=os.path.join(args.output_path, config.tag, run_id))

    # Setup Lightning

    if args.lightning:
        # TODO: Bundle this inside the agent, alongside the rest of the device mapping code. Need to pass the Fabric obj inside somehow...
        lightning.fabric.utilities.seed.seed_everything(config.get("seed", 123))

        fabric = lightning.fabric.Fabric(
            accelerator="cpu" if args.cpu else "cuda",
            devices=torch.cuda.device_count(),
            strategy="ddp",
            # precision=("32-true" if args.no_amp else "bf16-mixed"),
        )
        fabric.launch()

        if args.train:
            # for optimizer in agent.optimizers.optimizers:
            #     agent.model, optimizer = fabric.setup(agent.model, optimizer)
            agent.model, *agent.optimizers.optimizers = fabric.setup(agent.model, *agent.optimizers.optimizers)

        agent.backward = fabric.backward

        if data_loader._train.exists:
            data_loader.train_loader = fabric.setup_dataloaders(data_loader.train_loader)
        if data_loader._valid.exists:
            data_loader.valid_loader = fabric.setup_dataloaders(data_loader.valid_loader)
        if data_loader._test.exists:
            data_loader.test_loader = fabric.setup_dataloaders(data_loader.test_loader)
        use_lightning = True
    else:
        logger.info("Disabling Lightning!")
        use_lightning = False

    if args.load_chkpt is not None:
        logger.info("Loading from checkpoint:")
        logger.info(args.load_chkpt)
        agent.load_checkpoint(args.load_chkpt, write_pointer=not args.copy_chkpt)
        logger.info("...loaded!")
        if args.copy_chkpt:
            logger.info("Saving a local checkpoint copy")
            agent.save_checkpoint()

    if args.train:
        if data_loader.train_loader is None:
            raise Exception("Selected dataset does not include a training split - cannot train!")
        logger.info("Starting training...")

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

        agent.train(data_loader, use_lightning=use_lightning)
        logger.info("...training done!")

        if args.reload_after_train:
            logger.info("Training done - reloading saved model")
            torch.cuda.empty_cache()
            save_path = os.path.join(agent.run_output_path, "model", "checkpoint.pt")
            agent.load_checkpoint(save_path, write_pointer=False)
            logger.info("...loaded!")

    if args.validate_train:
        if data_loader.train_loader is None:
            raise Exception("Selected dataset does not include a training split - cannot train!")

        set_status_mckenzie("validating")

        logger.info("Starting validation (on training set)...")
        _ = agent.validate(
            data_loader,
            force_save_output=True,
            use_train=True,
            save_model=False,
            slow_metrics=True,
            use_lightning=use_lightning,
        )
        logger.info("...validation done!")

    if args.validate:
        if data_loader.valid_loader is None:
            raise Exception("Selected dataset does not include a dev split - cannot run validation!")

        # TEMP: save out the VQ embeds *after* they've been trained, for debug
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
                    agent.run_output_path + "/vqembeds_post.pt",
                )

        set_status_mckenzie("validating")

        logger.info("Starting validation...")
        _ = agent.validate(
            data_loader, force_save_output=True, save_model=False, slow_metrics=True, use_lightning=use_lightning
        )
        logger.info("...validation done!")

    if args.test:
        if data_loader.test_loader is None:
            raise Exception("Selected dataset does not include a test split - cannot run test evaluation!")

        set_status_mckenzie("validating")

        logger.info("Starting testing...")
        _ = agent.validate(
            data_loader,
            force_save_output=True,
            use_test=True,
            save_model=False,
            slow_metrics=True,
            use_lightning=use_lightning,
        )
        logger.info("...testing done!")


if __name__ == "__main__":
    main()
