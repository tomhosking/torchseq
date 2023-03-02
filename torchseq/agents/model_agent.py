import json
import os

import numpy as np
import torch
import torch.optim as optim


from collections import defaultdict
from tqdm import tqdm

from torchseq.agents.base import BaseAgent

from torchseq.models.lr_schedule import get_scheduler
from torchseq.models.samplers.beam_search import BeamSearchSampler
from torchseq.models.samplers.diverse_beam import DiverseBeamSearchSampler
from torchseq.models.samplers.greedy import GreedySampler
from torchseq.models.samplers.parallel_nucleus import ParallelNucleusSampler
from torchseq.models.rerankers.qa_reranker import QaReranker
from torchseq.models.rerankers.topk import TopkReducer
from torchseq.models.rerankers.ngram_reranker import NgramReranker
from torchseq.models.rerankers.backtranslate_reranker import BacktranslateReranker
from torchseq.models.rerankers.combo import CombinationReranker
from torchseq.models.samplers.teacher_force import TeacherForcedSampler
from torchseq.utils.logging import Logger
from torchseq.utils.mckenzie import update_mckenzie
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.cache import Cache
from torchseq.utils.optimizer_group import OptimizerGroup, SchedulerGroup
from torchseq.utils.wandb import wandb_log

from torchseq.models.ranger import Ranger
from torchseq.utils.loss_dropper import LossDropper
from torchseq.utils.seed import set_seed

from torchseq.metric_hooks.qg_metric import QGMetricHook
from torchseq.metric_hooks.textual import TextualMetricHook
from torchseq.metric_hooks.default import DefaultMetricHook
from torchseq.metric_hooks.sep_ae import SepAEMetricHook
from torchseq.metric_hooks.hrq_agg import HRQAggregationMetricHook
from torchseq.metric_hooks.rouge import RougeMetricHook
from torchseq.metric_hooks.semparse import SemanticParsingMetricHook


# Variable length sequences = worse performance if we try to optimise
from torch.backends import cudnn

cudnn.benchmark = False


class ModelAgent(BaseAgent):
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
    ):
        """
        Main constructor for an Agent
        """
        super().__init__(config)

        self.cache = None
        self.data_path = data_path
        self.run_id = run_id
        self.silent = silent
        self.verbose = verbose

        self.training_mode = training_mode
        self.run_output_path = None

        set_seed(config.get("seed", 123))

        self.input_tokenizer = Tokenizer(config.prepro.get_first(["input_tokenizer", "tokenizer"]), data_path)
        self.output_tokenizer = Tokenizer(config.prepro.get_first(["output_tokenizer", "tokenizer"]), data_path)

        # Slightly hacky way of allowing for inference-only use
        if run_id is not None:
            self.run_output_path = os.path.join(output_path, self.config.tag, self.run_id)
            if os.path.exists(self.run_output_path):
                self.logger.warning(
                    "Output path ({:}) already exists! Files may be overwritten".format(self.run_output_path)
                )
            else:
                os.makedirs(self.run_output_path)
                with open(os.path.join(self.run_output_path, "config.json"), "w") as f:
                    json.dump(config.data, f, indent=4)

            Logger(log_path=self.run_output_path + "/logs", interval=config.training.get("log_interval", 100))

            self.cache = Cache(cache_root if cache_root is not None else self.run_output_path)

        if self.config.training.get("loss_dropping", 0) > 0:
            self.dropper = LossDropper(dropc=self.config.training.get("loss_dropping", 0), recompute=10000)

        # initialize counter
        self.best_metric = None
        self.all_metrics_at_best = {}
        self.best_epoch = None
        self.current_epoch = 0
        self.global_step = 0
        self.global_step = 0

    def create_optimizer(self):
        """
        Initialise the optimizer this agent will use
        """
        param_group_config = self.config.training.optimizer.get("param_groups", {})
        optimizer_list = []
        scheduler_list = []
        if len(param_group_config) > 0:
            for pattern, cfg in param_group_config.items():
                params = [p for n, p in self.model.named_parameters() if p.requires_grad and pattern in n]
                if len(params) == 0:
                    raise Exception("Optimizer group {:} didnt match any model parameters!".format(pattern))

                curr_lr = cfg.get("lr", self.config.training.optimizer.lr)
                curr_bsz = cfg.get("optim_batch_size", self.config.training.optim_batch_size)
                curr_scheduled = cfg.get("lr_schedule", self.config.training.optimizer.lr_schedule)
                curr_group = [{"params": params, "lr": curr_lr}]

                # Adjust LR for different batch sizes
                curr_lr *= float(self.config.training.optim_batch_size) / float(curr_bsz)

                if self.config.training.optimizer.type == "adam":
                    betas = cfg.get(
                        "betas", (self.config.training.optimizer.beta1, self.config.training.optimizer.beta2)
                    )
                    curr_optimizer = optim.Adam(
                        curr_group,
                        lr=curr_lr,
                        betas=betas,
                        eps=1e-9,
                    )
                elif self.config.training.optimizer.type == "sgd":
                    curr_optimizer = optim.SGD(curr_group, lr=curr_lr)
                elif self.config.training.optimizer.type == "ranger":
                    curr_optimizer = Ranger(curr_group)
                else:
                    raise Exception("Unrecognised optimiser: " + self.config.training.optimizer.type)

                curr_optimizer.optim_batch_size = curr_bsz

                curr_scheduler = get_scheduler(
                    curr_optimizer,
                    base_lr=curr_lr,
                    scheduled=curr_scheduled,
                    warmup=self.config.training.optimizer.get("lr_warmup_steps", 10000) > 0,
                    num_warmup_steps=self.config.training.optimizer.get("lr_warmup_steps", 10000),
                    legacy=self.config.training.optimizer.get(
                        "lr_schedule_legacy", True
                    ),  # TODO: deprecate this and change the default for v3
                )

                optimizer_list.append(curr_optimizer)
                scheduler_list.append(curr_scheduler)

            # Add the remaining parameters as a default group
            remaining_params = [
                p
                for n, p in self.model.named_parameters()
                if p.requires_grad and sum([1 if p in n else 0 for p in param_group_config.keys()]) == 0
            ]
            def_param_group = [{"params": remaining_params}]
        else:
            def_param_group = [p for p in self.model.parameters() if p.requires_grad]

        if self.config.training.optimizer.type == "adam":
            def_optimizer = optim.Adam(
                def_param_group,
                lr=self.config.training.optimizer.lr,
                betas=(self.config.training.optimizer.beta1, self.config.training.optimizer.beta2),
                eps=1e-9,
            )
        elif self.config.training.optimizer.type == "sgd":
            def_optimizer = optim.SGD(def_param_group, lr=self.config.training.optimizer.lr)
        elif self.config.training.optimizer.type == "ranger":
            def_optimizer = Ranger(def_param_group)
        else:
            raise Exception("Unrecognised optimiser: " + self.config.training.optimizer.type)

        def_optimizer.optim_batch_size = self.config.training.optim_batch_size

        def_scheduler = get_scheduler(
            def_optimizer,
            base_lr=self.config.training.optimizer.lr,
            scheduled=self.config.training.optimizer.lr_schedule,
            warmup=self.config.training.optimizer.get("lr_warmup_steps", 10000) > 0,
            num_warmup_steps=self.config.training.optimizer.get("lr_warmup_steps", 10000),
            legacy=self.config.training.optimizer.get("lr_schedule_legacy", True),
        )

        optimizer_list.append(def_optimizer)
        scheduler_list.append(def_scheduler)

        self.optimizers = OptimizerGroup(optimizer_list)
        self.schedulers = SchedulerGroup(scheduler_list)

    def create_samplers(self):
        """
        Initialise sampling methods for this agent
        """
        self.decode_greedy = GreedySampler(self.config, self.output_tokenizer, self.device)
        self.decode_beam = BeamSearchSampler(self.config, self.output_tokenizer, self.device)
        self.decode_dbs = DiverseBeamSearchSampler(self.config, self.output_tokenizer, self.device)
        self.decode_teacher_force = TeacherForcedSampler(self.config, self.output_tokenizer, self.device)
        self.decode_nucleus = ParallelNucleusSampler(self.config, self.output_tokenizer, self.device)

        if self.config.data.get("reranker", None) is not None:
            if self.config.reranker.data.get("strategy", None) == "qa":
                self.reranker = QaReranker(self.config, self.output_tokenizer, self.device)
            elif self.config.reranker.data.get("strategy", None) == "ngram":
                self.reranker = NgramReranker(self.config, self.device, self.src_field)
            elif self.config.reranker.data.get("strategy", None) == "backtranslate":
                self.reranker = BacktranslateReranker(
                    self.config, self.output_tokenizer.pad_id, self.device, self.src_field, self.model
                )
            elif self.config.reranker.data.get("strategy", None) == "combo":
                self.reranker = CombinationReranker(
                    self.config, self.output_tokenizer, self.device, self.src_field, self.model
                )
        else:
            self.reranker = TopkReducer(self.config, self.output_tokenizer.pad_id, self.device)

    def load_checkpoint(self, file_name, write_pointer=True):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        self.logger.info("Loading from checkpoint " + file_name)
        checkpoint = torch.load(file_name, map_location=(None if torch.cuda.is_available() else "cpu"))
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if len(missing_keys) > 0:
            self.logger.warning(
                "Some keys were missing from the loaded checkpoint: \n{:}".format("\n".join(missing_keys))
            )
        if len(unexpected_keys) > 0:
            self.logger.warning(
                "Some unexpected keys were found in the loaded checkpoint: \n{:}".format("\n".join(unexpected_keys))
            )

        if self.training_mode:
            if "optimizer_state_dict" in checkpoint:
                self.optimizers.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.schedulers.load_state_dict(checkpoint["scheduler_state_dict"])

            if self.config.training.optimizer.type == "sgd":
                # Insert meta params if not there already
                for opt in self.optimizers:
                    for param_group in opt.param_groups:
                        if "momentum" not in param_group:
                            param_group["momentum"] = 0
                        if "dampening" not in param_group:
                            param_group["dampening"] = 0

        # self.current_epoch = checkpoint['epoch']
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
        else:
            self.logger.warning("Checkpoint is missing global_step value!")

        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]

        # self.loss = checkpoint["loss"]
        self.best_metric = (
            checkpoint["best_metric"]
            if ("reset_metrics" not in self.config.training.data or not self.config.training.reset_metrics)
            else None
        )

        if write_pointer and self.run_id is not None:
            pointer_filepath = os.path.join(self.run_output_path, "orig_model.txt")

            with open(pointer_filepath, "w") as f:
                f.write(file_name)

    def save_checkpoint(self, file_name="checkpoint.pt"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        save_path = os.path.join(self.run_output_path, "model")
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizers.state_dict() if self.training_mode else None,
                "scheduler_state_dict": self.schedulers.state_dict() if self.training_mode else None,
                # "loss": self.loss,
                "best_metric": self.best_metric,
            },
            os.path.join(save_path, file_name),
        )

    def begin_epoch_hook(self):
        """
        Override this if you want any logic to be triggered at the start of a new training epoch
        """
        pass

    def text_to_batch(self, input):
        """
        Convert a dictionary of strings to a batch that can be used as input to the model
        """
        raise NotImplementedError("Your model is missing a text_to_batch method!")

    def step_train(self, batch, tgt_field) -> torch.Tensor:
        """
        Run a single training step
        """
        batch["_global_step"] = self.global_step

        output, logits, _, memory = self.decode_teacher_force(self.model, batch, tgt_field)

        this_loss = torch.zeros(output.shape[0], dtype=torch.float).to(self.device)

        if self.config.training.get("xe_loss", True):
            elementwise_loss = self.loss(logits.permute(0, 2, 1), batch[tgt_field])
            if elementwise_loss.isnan_().any():
                self.logger.error("NaN loss detected! Aborting training")
            this_loss += elementwise_loss[:, 1:].sum(dim=1) / (batch[tgt_field + "_len"] - 1).to(this_loss)

        if "loss" in memory:
            this_loss += memory["loss"]

        if self.config.training.get("loss_dropping", 0) > 0:
            mask = self.dropper(this_loss)  # The dropper returns a mask of 0s where data should be dropped.
            this_loss *= mask  # Mask out the high losses')

        return this_loss

    def train(self, data_loader) -> None:
        """
        Main training loop
        :return:
        """
        if self.tgt_field is None:
            raise Exception("You need to specify the target output field! ie which element of a batch is the output")

        if data_loader is None:
            raise Exception(
                "Agent was created with a null dataset - you can only use this for on-the-fly inference, not training!"
            )

        update_mckenzie(0, "-")
        wandb_log({"progress": self.current_epoch / self.config.training.num_epochs * 100}, step=self.global_step)

        epochs_without_improvement = 0
        best_loss = 1e10

        # If we're starting above zero, means we've loaded from chkpt -> validate to give a starting point for fine tuning
        if self.global_step > 0:
            self.begin_epoch_hook()
            test_loss, best_metrics, _, _ = self.validate(data_loader, save_model=True, training_loop=True)

            best_loss = test_loss

            self.logger.info("Validation: Average loss: {:.4f}".format(test_loss))

            Logger().log_scalar("dev/loss", test_loss, self.global_step)

        for epoch in range(self.config.training.num_epochs):
            self.begin_epoch_hook()

            self.train_one_epoch(data_loader)

            self.current_epoch += 1

            if self.current_epoch > self.config.training.warmup_epochs:
                test_loss, best_metrics, _, _ = self.validate(data_loader, save_model=True, training_loop=True)
                self.logger.info("Validation: Average loss: {:.4f}".format(test_loss))

                Logger().log_scalar("dev/loss", test_loss, self.global_step)
                if test_loss < best_loss:
                    best_loss = test_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement > self.config.training.get("early_stopping_patience", 3):
                    self.logger.info(
                        "No improvement in dev loss for {:} epochs - stopping early".format(
                            self.config.training.get("early_stopping_patience", 3)
                        )
                    )
                    break
            else:
                # We won't have metrics - but we should update the progress tracker
                self.update_dashboard()

        self.logger.info("## Training completed {:} epochs".format(self.current_epoch + 1))
        self.logger.info("## Best metrics: {:}".format(self.all_metrics_at_best))

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

        pbar = tqdm(data_loader.train_loader, desc="Epoch {:}".format(self.current_epoch + 1), disable=self.silent)

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
                # Trigger if the threshold is exceeded OR if this is the last (partial) batch
                if steps >= opt.optim_batch_size or batch_idx == len(pbar) - 1:
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

                    with torch.inference_mode():
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

    def step_validate(self, batch, tgt_field, sample_outputs=True, calculate_loss=True, reduce_outputs=True):
        """
        Perform a single inference step
        """

        batch["_global_step"] = self.global_step

        if not sample_outputs:
            dev_output = None
            dev_output_lens = None
            dev_scores = None
        elif self.config.eval.sampler == "nucleus":
            beam_output, beam_scores, beam_lens, _ = self.decode_nucleus(self.model, batch, tgt_field)

        elif self.config.eval.sampler == "beam":
            beam_output, beam_scores, beam_lens, _ = self.decode_beam(self.model, batch, tgt_field)

        elif self.config.eval.sampler == "diverse_beam":
            beam_output, beam_scores, beam_lens, _ = self.decode_dbs(self.model, batch, tgt_field)

        elif self.config.eval.sampler == "greedy":
            greedy_output, greedy_scores, greedy_lens, _ = self.decode_greedy(self.model, batch, tgt_field)
            # Greedy returns a single estimate, so expand to create a fake "beam"
            beam_output = greedy_output.unsqueeze(1)
            beam_scores = greedy_scores.unsqueeze(1)
            beam_lens = greedy_lens.unsqueeze(1)
        else:
            raise Exception("Unknown sampling method!")

        # Rerank (or just select top-1)
        if sample_outputs and reduce_outputs:
            dev_output, dev_output_lens, dev_scores = self.reranker(
                beam_output, beam_lens, batch, tgt_field, scores=beam_scores, sort=True
            )
        elif sample_outputs:
            dev_output, dev_output_lens, dev_scores = self.reranker(
                beam_output, beam_lens, batch, tgt_field, scores=beam_scores, sort=True, top1=False
            )

        if calculate_loss:
            _, logits, _, memory = self.decode_teacher_force(self.model, batch, tgt_field)
            this_loss = self.loss(logits.permute(0, 2, 1), batch[tgt_field])

            normed_loss = torch.sum(this_loss[:, 1:], dim=1) / (batch[tgt_field + "_len"] - 1).to(this_loss)

            if "loss" in memory:
                normed_loss += memory["loss"]

            normed_loss = torch.mean(normed_loss, dim=0)

            if "vq_codes" in memory:
                for h_ix in range(memory["vq_codes"].shape[1]):
                    self.vq_codes[h_ix].extend(memory["vq_codes"][:, h_ix].tolist())

        else:
            logits = None
            normed_loss = None
            memory = None
        return normed_loss, dev_output, dev_output_lens, dev_scores, logits, memory

    def inference(
        self, data_loader, memory_keys_to_return=None, metric_hooks=[], use_test=False, training_loop=False, desc=None
    ):
        """
        Inner inference loop - generate outputs, but don't run metrics. This is the recommended method for running inference from a script.
        """
        test_loss = 0

        pred_output = []
        gold_output = []
        gold_input = []  # needed for SARI

        memory_values_to_return = defaultdict(lambda: [])

        self.vq_codes = defaultdict(lambda: [])

        self.model.eval()

        for hook in metric_hooks:
            hook.on_begin_epoch(use_test)

        desc = "Inference" if desc is None else desc

        with torch.inference_mode():
            num_samples = 0
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=desc, disable=self.silent)):
                batch = {k: (v.to(self.device) if k[-5:] != "_text" and k[0] != "_" else v) for k, v in batch.items()}

                curr_batch_size = batch[[k for k in batch.keys() if k[-5:] != "_text"][0]].size()[0]

                this_loss, dev_output, dev_output_lens, dev_scores, logits, memory = self.step_validate(
                    batch,
                    self.tgt_field,
                    sample_outputs=self.config.eval.data.get("sample_outputs", True) and not training_loop,
                    reduce_outputs=(self.config.eval.data.get("topk", 1) == 1),
                )

                test_loss += this_loss * curr_batch_size

                if memory_keys_to_return is not None:
                    for mem_key in memory_keys_to_return:
                        memory_values_to_return[mem_key].append(memory[mem_key].cpu().detach())

                num_samples += curr_batch_size

                #  Handle top-1 sampling
                if (
                    self.config.eval.data.get("sample_outputs", True)
                    and not training_loop
                    and (self.config.eval.data.get("topk", 1) == 1)
                ):
                    for ix, pred in enumerate(dev_output.data):
                        pred_output.append(self.output_tokenizer.decode(pred[: dev_output_lens[ix]]))
                    for ix, gold in enumerate(batch[self.tgt_field]):
                        # gold_output.append(self.output_tokenizer.decode(gold[: batch[self.tgt_field + "_len"][ix]]))
                        gold_output.append(batch[self.tgt_field + "_text"][ix])
                    for ix, gold in enumerate(batch[self.src_field]):
                        # gold_input.append(self.input_tokenizer.decode(gold[: batch[self.src_field + "_len"][ix]]))
                        gold_input.append(batch[self.src_field + "_text"][ix])

                    if batch_idx % 200 == 0 and self.verbose:
                        # print(gold_input[-2:])
                        self.logger.info(gold_output[-2:])
                        self.logger.info(pred_output[-2:])

                # Handle top-k sampling
                if (
                    self.config.eval.data.get("sample_outputs", True)
                    and not training_loop
                    and not (self.config.eval.data.get("topk", 1) == 1)
                ):
                    topk = self.config.eval.data.get("topk", 1)
                    for ix, pred in enumerate(dev_output.data):
                        pred_output.append(
                            [
                                self.output_tokenizer.decode(pred[jx][: dev_output_lens[ix][jx]])
                                for jx in range(min(len(pred), topk))
                            ]
                        )
                    for ix, gold in enumerate(batch[self.tgt_field]):
                        # gold_output.append(self.output_tokenizer.decode(gold[: batch[self.tgt_field + "_len"][ix]]))
                        gold_output.append(batch[self.tgt_field + "_text"][ix])

                    # if batch_idx > 20:
                    #     break

                for hook in metric_hooks:
                    # This is a horrible way of getting the current batch worth of decoded output - tidy it up at some point!
                    hook.on_batch(batch, logits, pred_output[-curr_batch_size:], memory, use_test)

        test_loss = test_loss / num_samples

        all_metrics = {}
        for hook in metric_hooks:
            hook_values = hook.on_end_epoch(self, use_test)
            all_metrics = {**all_metrics, **hook_values}

        if memory_keys_to_return is not None:
            for mem_key in memory_keys_to_return:
                memory_values_to_return[mem_key] = torch.cat(memory_values_to_return[mem_key], dim=0)

        return test_loss, all_metrics, (pred_output, gold_output, gold_input), memory_values_to_return

    def validate(
        self,
        data_loader,
        force_save_output=False,
        use_test=False,
        use_train=False,
        save_model=True,
        memory_keys_to_return=None,
        slow_metrics=False,
        training_loop=False,
    ):
        """
        One cycle of model validation. This includes metrics, and is the entry point for eval using the CLI.
        :return:
        """

        if self.tgt_field is None:
            raise Exception("You need to specify the target output field! ie which element of a batch is the output")

        self.logger.info("## Validating after {:} epochs".format(self.current_epoch))
        self.model.eval()

        slow_metrics = slow_metrics and self.config.eval.data.get("sample_outputs", True) and not training_loop

        # Register the metric hooks to be used
        metric_hooks = [DefaultMetricHook(self.config, self.output_tokenizer, self.src_field, self.tgt_field)]

        if self.config.eval.data.get("sample_outputs", True) and not training_loop:
            metric_hooks += [TextualMetricHook(self.config, self.output_tokenizer, self.src_field, self.tgt_field)]

        if slow_metrics:
            metric_hooks += [QGMetricHook(self.config, self.output_tokenizer, self.src_field, self.tgt_field)]

        if slow_metrics and "sep_ae" in self.config.eval.get("metrics", {}).keys():
            metric_hooks += [SepAEMetricHook(self.config, self.output_tokenizer, self.src_field, self.tgt_field)]

        if slow_metrics and "semparse" in self.config.eval.get("metrics", {}).keys():
            metric_hooks += [
                SemanticParsingMetricHook(self.config, self.output_tokenizer, self.src_field, self.tgt_field)
            ]

        if slow_metrics and "hrq_agg" in self.config.eval.get("metrics", {}).keys():
            metric_hooks += [
                HRQAggregationMetricHook(self.config, self.output_tokenizer, self.src_field, self.tgt_field)
            ]

        if (
            "rouge" in self.config.eval.get("metrics", {}).keys()
            and self.config.eval.data.get("sample_outputs", True)
            and not training_loop
        ):
            metric_hooks += [RougeMetricHook(self.config, self.output_tokenizer, self.src_field, self.tgt_field)]

        if use_test:
            split_slug = "test"
            self.logger.info("***** USING TEST SET ******")
            valid_loader = data_loader.test_loader
        elif use_train:
            split_slug = "train"
            self.logger.info("***** USING TRAINING SET ******")
            valid_loader = data_loader.train_loader
        else:
            split_slug = "dev"
            valid_loader = data_loader.valid_loader

        test_loss, all_metrics, (pred_output, gold_output, gold_input), memory_values_to_return = self.inference(
            valid_loader,
            memory_keys_to_return,
            metric_hooks,
            use_test,
            training_loop,
            desc="Validating after {:} epochs".format(self.current_epoch),
        )

        for h_ix, codes in self.vq_codes.items():
            if len(codes) > 0:
                Logger().log_histogram(f"vq_codes/{split_slug}/h{h_ix}", codes, self.global_step)

        # if len(self.vq_codes) > 0 and self.run_id is not None:
        #     with open(os.path.join(self.output_path, self.config.tag, self.run_id, "codes.json"), "w") as f:
        #         json.dump(self.vq_codes, f)

        # TODO: sort this out - there's got to be a more compact way of doing it all
        if (
            self.best_metric is None
            or test_loss < self.best_metric
            or force_save_output
            or (
                self.config.training.early_stopping_lag > 0
                and self.best_epoch is not None
                and (self.current_epoch - self.best_epoch) <= self.config.training.early_stopping_lag > 0
            )
        ):

            self.all_metrics_at_best = {"nll": test_loss.item(), "epoch": self.current_epoch, **all_metrics}

            wandb_log({split_slug + "/" + k: v for k, v in self.all_metrics_at_best.items()}, step=self.global_step)

            if self.run_id is not None:
                with open(os.path.join(self.run_output_path, f"output.{split_slug}.txt"), "w") as f:
                    f.write("\n".join([pred for pred in pred_output]))
                with open(os.path.join(self.run_output_path, f"metrics.{split_slug}.json"), "w") as f:
                    json.dump(self.all_metrics_at_best, f, indent=4)

        if (
            self.best_metric is None
            or test_loss < self.best_metric
            or (
                self.config.training.early_stopping_lag > 0
                and self.best_epoch is not None
                and (self.current_epoch - self.best_epoch) <= self.config.training.early_stopping_lag > 0
            )
        ):
            if "bleu" in self.all_metrics_at_best:
                self.logger.info("Current best BLEU: {:}".format(self.all_metrics_at_best["bleu"]))

            if self.best_metric is None:
                self.best_epoch = self.current_epoch
                self.best_metric = test_loss
            elif test_loss < self.best_metric:
                self.logger.info("New best score! Saving...")
                self.best_epoch = self.current_epoch
                self.best_metric = test_loss
            else:
                self.logger.info("Early stopping lag active: saving...")

            if save_model and self.run_id is not None:
                self.save_checkpoint()

        self.update_dashboard()

        return test_loss, self.all_metrics_at_best, pred_output, memory_values_to_return

    def update_dashboard(self):
        """
        Update McKenzie with the latest metric values
        """
        # wandb_log(
        #     {
        #         "bleu": self.all_metrics_at_best.get("bleu", None),
        #         "nll": self.all_metrics_at_best.get("nll", None),
        #     },
        #     step=self.global_step,
        # )
        wandb_log({"progress": self.current_epoch / self.config.training.num_epochs * 100}, step=self.global_step)

        if "bleu" in self.all_metrics_at_best:
            update_mckenzie(
                self.current_epoch / self.config.training.num_epochs * 100,
                "{:0.2f}".format(self.all_metrics_at_best[self.config.training.data.get("mckenzie_metric", "bleu")]),
            )

        elif "nll" in self.all_metrics_at_best:
            update_mckenzie(
                self.current_epoch / self.config.training.num_epochs * 100,
                "{:0.2f}".format(self.all_metrics_at_best["nll"]),
            )
        else:
            update_mckenzie(
                self.current_epoch / self.config.training.num_epochs * 100,
                "-",
            )
