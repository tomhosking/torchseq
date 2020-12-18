import json
import os

import numpy as np
import torch
import torch.optim as optim

from collections import defaultdict
from tqdm import tqdm

from torchseq.agents.base import BaseAgent

from torchseq.models.lr_schedule import get_lr
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
from torchseq.utils.metrics import bleu_corpus, meteor_corpus
from torchseq.utils.sari import SARIsent
import torchseq.utils.tokenizer as tokenizer
from torchseq.utils.tokenizer import Tokenizer


from torchseq.models.ranger import Ranger
from torchseq.utils.loss_dropper import LossDropper
from torchseq.utils.seed import set_seed

from torchseq.metric_hooks.qg_metric import QGMetricHook
from torchseq.metric_hooks.textual import TextualMetricHook
from torchseq.metric_hooks.default import DefaultMetricHook
from torchseq.metric_hooks.sep_ae import SepAEMetricHook


# Variable length sequences = worse performance if we try to optimise
from torch.backends import cudnn

cudnn.benchmark = False


class ModelAgent(BaseAgent):
    def __init__(self, config, run_id, output_path, silent=False, training_mode=True, verbose=True):
        super().__init__(config)

        self.run_id = run_id
        self.silent = silent
        self.verbose = verbose
        self.output_path = output_path
        self.training_mode = training_mode

        set_seed(config.get("seed", 123))

        # This is lovely and hacky isn't it. Could we maybe pass it in as an arg?
        tokenizer.DATA_PATH = config.env.data_path
        tokenizer.Tokenizer(config.prepro.tokenizer)

        # Slightly hacky way of allowing for inference-only use
        if run_id is not None:
            os.makedirs(os.path.join(self.output_path, self.config.tag, self.run_id))
            with open(os.path.join(self.output_path, self.config.tag, self.run_id, "config.json"), "w") as f:
                json.dump(config.data, f, indent=4)

            Logger(log_path=self.output_path + "/" + self.config.tag + "/" + self.run_id + "/logs")

        if self.config.training.get("loss_dropping", 0) > 0:
            self.dropper = LossDropper(dropc=self.config.training.get("loss_dropping", 0), recompute=5000)

        # initialize counter
        self.best_metric = None
        self.all_metrics_at_best = {}
        self.best_epoch = None
        self.current_epoch = 0
        self.global_step = 0
        self.global_step = 0

    def create_optimizer(self):
        if self.config.training.opt == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.lr,
                betas=(self.config.training.beta1, self.config.training.beta2),
                eps=1e-9,
            )
        elif self.config.training.opt == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.training.lr)
        elif self.config.training.opt == "ranger":
            self.optimizer = Ranger(self.model.parameters())

        else:
            raise Exception("Unrecognised optimiser: " + self.config.training.opt)

    def create_samplers(self):
        self.decode_greedy = GreedySampler(self.config, self.device)
        self.decode_beam = BeamSearchSampler(self.config, self.device)
        self.decode_dbs = DiverseBeamSearchSampler(self.config, self.device)
        self.decode_teacher_force = TeacherForcedSampler(self.config, self.device)
        self.decode_nucleus = ParallelNucleusSampler(self.config, self.device)

        if self.config.data.get("reranker", None) is not None:
            if self.config.reranker.data.get("strategy", None) == "qa":
                self.reranker = QaReranker(self.config, self.device)
            elif self.config.reranker.data.get("strategy", None) == "ngram":
                self.reranker = NgramReranker(self.config, self.device, self.src_field)
            elif self.config.reranker.data.get("strategy", None) == "backtranslate":
                self.reranker = BacktranslateReranker(self.config, self.device, self.src_field, self.model)
            elif self.config.reranker.data.get("strategy", None) == "combo":
                self.reranker = CombinationReranker(self.config, self.device, self.src_field, self.model)
        else:
            self.reranker = TopkReducer(self.config, self.device)

    def load_checkpoint(self, file_name, write_pointer=True):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        self.logger.info("Loading from checkpoint " + file_name)
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.training_mode:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # self.current_epoch = checkpoint['epoch']
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
        else:
            self.logger.warn("Checkpoint is missing global_step value!")
        self.loss = checkpoint["loss"]
        self.best_metric = (
            checkpoint["best_metric"]
            if ("reset_metrics" not in self.config.training.data or not self.config.training.reset_metrics)
            else None
        )

        if self.config.training.opt == "sgd":
            # Insert meta params if not there already
            for param_group in self.optimizer.param_groups:
                if "momentum" not in param_group:
                    param_group["momentum"] = 0
                if "dampening" not in param_group:
                    param_group["dampening"] = 0

        if write_pointer and self.run_id is not None:
            pointer_filepath = os.path.join(self.output_path, self.config.tag, self.run_id, "orig_model.txt")

            with open(pointer_filepath, "w") as f:
                f.write(file_name)

    def save_checkpoint(self, file_name="checkpoint.pt"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        save_path = os.path.join(self.output_path, self.config.tag, self.run_id, "model")
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
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

    def infer(self, input, reduce_outputs=True, sample_outputs=True, calculate_loss=False):
        """
        Run inference on a dictionary of raw inputs
        """
        batch = self.text_to_batch(input, self.device)

        _, output, output_lens, scores, logits, memory = self.step_validate(
            batch,
            tgt_field=self.tgt_field,
            sample_outputs=sample_outputs,
            calculate_loss=calculate_loss,
            reduce_outputs=reduce_outputs,
        )

        if sample_outputs and reduce_outputs:
            output_strings = [Tokenizer().decode(output.data[ix][: output_lens[ix]]) for ix in range(len(output_lens))]
        elif sample_outputs:
            # There's an extra dim of nesting
            output_strings = [
                [Tokenizer().decode(output.data[i][j][: output_lens[i][j]]) for j in range(len(output_lens[i]))]
                for i in range(len(output_lens))
            ]
        else:
            output_strings = None

        return output_strings, scores, memory

    def step_train(self, batch, tgt_field):
        batch["_global_step"] = self.global_step

        output, logits, _, memory = self.decode_teacher_force(self.model, batch, tgt_field)

        this_loss = torch.zeros(output.shape[0], dtype=torch.float).to(self.device)

        if self.config.training.get("xe_loss", True):
            this_loss += self.loss(logits.permute(0, 2, 1), batch[tgt_field]).sum(dim=1) / (
                batch[tgt_field + "_len"] - 1
            ).to(this_loss)

        if "loss" in memory:
            this_loss += memory["loss"]

        if self.config.training.get("loss_dropping", False):
            mask = self.dropper(this_loss)  # The dropper returns a mask of 0s where data should be dropped.
            this_loss *= mask  # Mask out the high losses')

        return this_loss

    def train(self, data_loader):
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

        # If we're starting above zero, means we've loaded from chkpt -> validate to give a starting point for fine tuning
        if self.global_step > 0:
            self.begin_epoch_hook()
            _ = self.validate(data_loader, save=True)

        for epoch in range(self.config.training.num_epochs):
            self.begin_epoch_hook()

            self.train_one_epoch(data_loader)

            self.current_epoch += 1

            if self.current_epoch > self.config.training.warmup_epochs:
                _ = self.validate(data_loader, save=True)
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

        self.model.train()

        self.logger.info("## Training epoch {:}".format(self.current_epoch))

        self.optimizer.zero_grad()
        steps_accum = 0

        start_step = self.global_step

        pbar = tqdm(data_loader.train_loader, desc="Epoch {:}".format(self.current_epoch), disable=self.silent)

        for batch_idx, batch in enumerate(pbar):
            batch = {k: (v.to(self.device) if k[-5:] != "_text" else v) for k, v in batch.items()}

            curr_batch_size = batch[[k for k in batch.keys() if k[-5:] != "_text"][0]].size()[0]

            # self.global_step += curr_batch_size

            # Weight the loss by the ratio of this batch to optimiser step size, so that LR is equivalent even when grad accumulation happens
            loss = (
                self.step_train(batch, self.tgt_field)
                * float(curr_batch_size)
                / float(self.config.training.optim_batch_size)
            )

            pbar.set_postfix(
                {"loss": loss.item() * float(self.config.training.optim_batch_size) / float(curr_batch_size)}
            )

            loss.backward()

            steps_accum += curr_batch_size

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_gradient)

            lr = get_lr(
                self.config.training.lr,
                self.global_step,
                self.config.training.lr_schedule,
                self.config.training.data.get("lr_warmup", True),
            )

            Logger().log_scalar("train/lr", lr, self.global_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient accumulation
            if steps_accum >= self.config.training.optim_batch_size:
                self.optimizer.step()
                self.optimizer.zero_grad()
                steps_accum = 0
                self.global_step += 1

            if batch_idx % self.config.training.log_interval == 0:
                # Loss is weighted for grad accumulation - unweight it for reporting
                Logger().log_scalar(
                    "train/loss",
                    loss * float(self.config.training.optim_batch_size) / float(curr_batch_size),
                    self.global_step,
                )

                # TODO: This is currently paraphrase specific! May work for other models but isn't guaranteed
                if batch_idx % (self.config.training.log_interval * 20) == 0 and self.verbose:

                    with torch.no_grad():
                        greedy_output, _, output_lens, _ = self.decode_greedy(self.model, batch, self.tgt_field)

                    self.logger.info(Tokenizer().decode(batch[self.src_field][0][: batch[self.src_field + "_len"][0]]))
                    self.logger.info(Tokenizer().decode(batch[self.tgt_field][0][: batch[self.tgt_field + "_len"][0]]))
                    self.logger.info(Tokenizer().decode(greedy_output.data[0][: output_lens[0]]))

                    # if self.config.encdec.data.get("variational", False):
                    #     print(self.model.mu)
                    #     print(self.model.logvar)

                torch.cuda.empty_cache()

            if (
                self.config.training.data.get("epoch_steps", 0) > 0
                and self.global_step - start_step >= self.config.training.epoch_steps
            ):
                self.logger.info(
                    "Epoch step size is set - validating after {:} training steps".format(
                        self.config.training.epoch_steps
                    )
                )
                break

    def step_validate(self, batch, tgt_field, sample_outputs=True, calculate_loss=True, reduce_outputs=True):

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
            normed_loss = torch.mean(
                torch.sum(this_loss, dim=1) / (batch[tgt_field + "_len"] - 1).to(this_loss), dim=0
            )

            if self.config.encdec.get("vector_quantized", False):
                for h_ix, vq_codes in enumerate(memory["vq_codes"]):
                    self.vq_codes[h_ix].extend(vq_codes.tolist())

        else:
            logits = None
            normed_loss = None
            memory = None
        return normed_loss, dev_output, dev_output_lens, dev_scores, logits, memory

    def validate(
        self,
        data_loader,
        save=False,
        force_save_output=False,
        use_test=False,
        use_train=False,
        save_model=True,
        memory_keys_to_return=None,
        slow_metrics=False,
    ):
        """
        One cycle of model validation
        :return:
        """

        if self.tgt_field is None:
            raise Exception("You need to specify the target output field! ie which element of a batch is the output")

        if data_loader is None:
            raise Exception(
                "Agent was created with a null dataset - you can only use this for on-the-fly inference, not validation!"
            )

        self.logger.info("## Validating after {:} epochs".format(self.current_epoch))
        self.model.eval()
        test_loss = 0

        # Register the metric hooks to be used
        metric_hooks = [DefaultMetricHook(self.config, self.src_field, self.tgt_field)]

        if self.config.eval.data.get("sample_outputs", True):
            metric_hooks += [TextualMetricHook(self.config, self.src_field, self.tgt_field)]

        if slow_metrics:
            metric_hooks += [QGMetricHook(self.config, self.src_field, self.tgt_field)]

        if slow_metrics and "sep_ae" in self.config.eval.get("metrics", []):
            metric_hooks += [SepAEMetricHook(self.config, self.src_field, self.tgt_field)]

        # TODO: add QA, LM metric for Qgen task
        # TODO: Add sem similarity for sep AE
        # TODO: Add weird BLEU variants for sep AE

        for hook in metric_hooks:
            hook.on_begin_epoch()

        pred_output = []
        gold_output = []
        gold_input = []  # needed for SARI

        self.vq_codes = defaultdict(lambda: [])

        memory_values_to_return = defaultdict(lambda: [])

        if use_test:
            self.logger.info("***** USING TEST SET ******")
            valid_loader = data_loader.test_loader
        elif use_train:
            self.logger.info("***** USING TRAINING SET ******")
            valid_loader = data_loader.train_loader
        else:
            valid_loader = data_loader.valid_loader

        with torch.no_grad():
            num_samples = 0
            for batch_idx, batch in enumerate(
                tqdm(valid_loader, desc="Validating after {:} epochs".format(self.current_epoch), disable=self.silent)
            ):
                batch = {k: (v.to(self.device) if k[-5:] != "_text" else v) for k, v in batch.items()}

                curr_batch_size = batch[[k for k in batch.keys() if k[-5:] != "_text"][0]].size()[0]

                this_loss, dev_output, dev_output_lens, dev_scores, logits, memory = self.step_validate(
                    batch,
                    self.tgt_field,
                    sample_outputs=self.config.eval.data.get("sample_outputs", True),
                    reduce_outputs=(self.config.eval.data.get("topk", 1) == 1),
                )

                test_loss += this_loss * curr_batch_size

                if memory_keys_to_return is not None:
                    for mem_key in memory_keys_to_return:
                        memory_values_to_return[mem_key].append(memory[mem_key].cpu().detach())

                num_samples += curr_batch_size

                #  Handle top-1 sampling
                if self.config.eval.data.get("sample_outputs", True) and (self.config.eval.data.get("topk", 1) == 1):
                    for ix, pred in enumerate(dev_output.data):
                        pred_output.append(Tokenizer().decode(pred[: dev_output_lens[ix]]))
                    for ix, gold in enumerate(batch[self.tgt_field]):
                        # gold_output.append(Tokenizer().decode(gold[: batch[self.tgt_field + "_len"][ix]]))
                        gold_output.append(batch[self.tgt_field + "_text"][ix])
                    for ix, gold in enumerate(batch[self.src_field]):
                        # gold_input.append(Tokenizer().decode(gold[: batch[self.src_field + "_len"][ix]]))
                        gold_input.append(batch[self.src_field + "_text"][ix])

                    if batch_idx % 200 == 0 and self.verbose:
                        # print(gold_input[-2:])
                        self.logger.info(gold_output[-2:])
                        self.logger.info(pred_output[-2:])

                # Handle top-k sampling
                if self.config.eval.data.get("sample_outputs", True) and not (
                    self.config.eval.data.get("topk", 1) == 1
                ):
                    topk = self.config.eval.data.get("topk", 1)
                    for ix, pred in enumerate(dev_output.data):
                        pred_output.append(
                            [
                                Tokenizer().decode(pred[jx][: dev_output_lens[ix][jx]])
                                for jx in range(min(len(pred), topk))
                            ]
                        )
                    for ix, gold in enumerate(batch[self.tgt_field]):
                        # gold_output.append(Tokenizer().decode(gold[: batch[self.tgt_field + "_len"][ix]]))
                        gold_output.append(batch[self.tgt_field + "_text"][ix])

                    # if batch_idx > 20:
                    #     break

                for hook in metric_hooks:
                    # This is a horrible way of getting the current batch worth of decoded output - tidy it up at some point!
                    hook.on_batch(batch, logits, pred_output[-curr_batch_size:], memory)

            test_loss /= num_samples
            self.logger.info("Dev set: Average loss: {:.4f}".format(test_loss))

        Logger().log_scalar("dev/loss", test_loss, self.global_step)

        all_metrics = {}
        for hook in metric_hooks:
            hook_values = hook.on_end_epoch(self)
            all_metrics = {**all_metrics, **hook_values}

        for h_ix, codes in self.vq_codes.items():
            Logger().log_histogram("vq_codes/h" + str(h_ix), torch.Tensor(codes), self.global_step)

        if len(self.vq_codes) > 0 and self.run_id is not None:
            with open(os.path.join(self.output_path, self.config.tag, self.run_id, "codes.json"), "w") as f:
                json.dump(self.vq_codes, f)

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

            self.all_metrics_at_best = {"nll": test_loss.item(), **all_metrics}

            if self.run_id is not None:
                with open(os.path.join(self.output_path, self.config.tag, self.run_id, "output.txt"), "w") as f:
                    f.write("\n".join([json.dumps(pred) for pred in pred_output]))
                with open(os.path.join(self.output_path, self.config.tag, self.run_id, "metrics.json"), "w") as f:
                    json.dump(self.all_metrics_at_best, f)

        if (
            self.best_metric is None
            or test_loss < self.best_metric
            or (
                self.config.training.early_stopping_lag > 0
                and self.best_epoch is not None
                and (self.current_epoch - self.best_epoch) <= self.config.training.early_stopping_lag > 0
            )
        ):

            if self.best_metric is None:
                self.best_epoch = self.current_epoch
                self.best_metric = test_loss
            elif test_loss < self.best_metric:
                self.logger.info("New best score! Saving...")
                self.best_epoch = self.current_epoch
                self.best_metric = test_loss
            else:
                self.logger.info("Early stopping lag active: saving...")

            if save_model:
                self.save_checkpoint()

        self.update_dashboard()

        if memory_keys_to_return is not None:
            for mem_key in memory_keys_to_return:
                memory_values_to_return[mem_key] = torch.cat(memory_values_to_return[mem_key], dim=0)

        return test_loss, self.all_metrics_at_best, pred_output, memory_values_to_return

    def update_dashboard(self):
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
