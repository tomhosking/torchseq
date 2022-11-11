import json
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


from collections import defaultdict
from tqdm import tqdm

from torchseq.agents.model_agent import ModelAgent
from torchseq.utils.logging import Logger

from torchseq.models.bottleneck_autoencoder import BottleneckAutoencoderModel
from torchseq.models.pretrained_adapter import PretrainedAdapterModel


"""
MetaLearning agent - allow for custom training loops, with more complex interactions between multiple dataloaders
"""


class MetaLearningAgent(ModelAgent):
    def __init__(
        self,
        config,
        run_id,
        output_path,
        data_path,
        silent=False,
        training_mode=True,
        verbose=True,
        cache_root=None,
        use_cuda=True,
    ):
        """
        Main constructor for an Agent
        """
        super().__init__(config, run_id, output_path, data_path, silent, training_mode, verbose, cache_root)

        self.tgt_field = "target"
        self.src_field = "source"

        # define loss
        self.loss = nn.CrossEntropyLoss(
            ignore_index=self.output_tokenizer.pad_id,
            reduction="none",
            label_smoothing=self.config.training.get("label_smoothing", 0.0),
        )

        # define model
        if self.config.data.get("model", None) is not None and self.config.model == "pretrained_adapter":
            self.model = PretrainedAdapterModel(
                self.config,
                self.input_tokenizer,
                self.output_tokenizer,
                src_field=self.src_field,
                tgt_field=self.tgt_field,
            )
        else:
            self.model = BottleneckAutoencoderModel(
                self.config, self.input_tokenizer, self.output_tokenizer, src_field=self.src_field
            )

        # define optimizer
        if training_mode:
            self.create_optimizer()

        self.set_device(use_cuda)

        self.create_samplers()

    def step_train(self, batch, tgt_field):
        """
        Run a single training step
        """
        batch["_global_step"] = self.global_step

        output, logits, _, memory = self.decode_teacher_force(self.model, batch, tgt_field)

        this_loss = torch.zeros(output.shape[0], dtype=torch.float).to(self.device)

        if self.config.training.get("xe_loss", True):
            elementwise_loss = self.loss(logits.permute(0, 2, 1), batch[tgt_field])
            this_loss += elementwise_loss[:, 1:].sum(dim=1) / (batch[tgt_field + "_len"] - 1).to(this_loss)

        if "loss" in memory:
            this_loss += memory["loss"]

        if self.config.training.get("loss_dropping", False):
            mask = self.dropper(this_loss)  # The dropper returns a mask of 0s where data should be dropped.
            this_loss *= mask  # Mask out the high losses')

        return this_loss

    def train_one_epoch(self, data_loader):
        """
        One epoch of training
        :return:
        """
        self.logger.info("## Training epoch {:}".format(self.current_epoch + 1))

        self.model.train()
        self.optimizers.zero_grad()

        # Keep track of how many steps have accumulated for each optimizer
        steps_accum = [0 for _ in self.optimizers]

        start_step = self.global_step

        pbar = tqdm(data_loader.train_loader, desc="Epoch {:}".format(self.current_epoch), disable=self.silent)

        for batch_idx, batch in enumerate(pbar):
            # TRAIN STEP BEGINS
            batch = {k: (v.to(self.device) if k[-5:] != "_text" else v) for k, v in batch.items()}

            curr_batch_size = batch[[k for k in batch.keys() if k[-5:] != "_text"][0]].size()[0]

            loss = (
                self.step_train(batch, self.tgt_field)
                * float(curr_batch_size)
                / float(self.config.training.optim_batch_size)
            )
            if not self.silent:
                pbar.set_postfix(
                    {
                        "loss": loss.detach().item()
                        * float(self.config.training.optim_batch_size)
                        / float(curr_batch_size),
                        "lr": self.optimizers[-1].param_groups[-1]["lr"],
                    }
                )

            loss.backward()
            # self.backward(loss)

            steps_accum = [steps + curr_batch_size for steps in steps_accum]

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.get("clip_gradient", 1e6))

            # Gradient accumulation
            for ix, (opt, sched, steps) in enumerate(zip(self.optimizers, self.schedulers, steps_accum)):
                if steps >= opt.optim_batch_size:
                    opt.step()
                    opt.zero_grad()
                    sched.step()
                    steps_accum[ix] = 0
                    # If this is the default opt, update the global step param
                    if ix == len(self.optimizers) - 1:
                        self.global_step += 1

            # TRAIN STEP ENDS

            if self.global_step % self.config.training.log_interval == 0:
                # Loss is weighted for grad accumulation - unweight it for reporting
                Logger().log_scalar(
                    "train/loss",
                    loss * float(self.config.training.optim_batch_size) / float(curr_batch_size),
                    self.global_step,
                )
                Logger().log_scalar("train/lr", self.optimizers[-1].param_groups[-1]["lr"], self.global_step)

                # TODO: This is currently paraphrase specific! May work for other models but isn't guaranteed
                if batch_idx % (self.config.training.log_interval * 20) == 0 and self.verbose:

                    with torch.no_grad():
                        greedy_output, _, output_lens, _ = self.decode_greedy(self.model, batch, self.tgt_field)

                    self.logger.info(
                        self.output_tokenizer.decode(batch[self.src_field][0][: batch[self.src_field + "_len"][0]])
                    )
                    self.logger.info(
                        self.output_tokenizer.decode(batch[self.tgt_field][0][: batch[self.tgt_field + "_len"][0]])
                    )
                    self.logger.info(self.output_tokenizer.decode(greedy_output.data[0][: output_lens[0]]))

                # torch.cuda.empty_cache()

            if (
                self.config.training.get("epoch_steps", 0) > 0
                and self.global_step - start_step >= self.config.training.epoch_steps
            ):
                self.logger.info(
                    "Epoch step size is set - validating after {:} training steps".format(
                        self.config.training.epoch_steps
                    )
                )
                break

        # Is this needed? empty cache at end of training loop just in case
        torch.cuda.empty_cache()
