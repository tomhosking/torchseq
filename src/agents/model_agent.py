import json
import os

import numpy as np
import torch
import torch.optim as optim
# Variable length sequences = worse performance if we try to optimise
from torch.backends import cudnn
from tqdm import tqdm

from agents.base import BaseAgent

from models.lr_schedule import get_lr
from models.samplers.beam_search import BeamSearchSampler
from models.samplers.greedy import GreedySampler
from models.samplers.parallel_nucleus import ParallelNucleusSampler
from models.samplers.reranking_reducer import RerankingReducer
from models.samplers.teacher_force import TeacherForcedSampler
from utils.logging import add_to_log
from utils.mckenzie import update_mckenzie
from utils.metrics import bleu_corpus
from utils.sari import SARIsent
from utils.tokenizer import BPE

from models.ranger import Ranger


# Variable length sequences = worse performance if we try to optimise
from torch.backends import cudnn
cudnn.benchmark = False


class ModelAgent(BaseAgent):
    def __init__(self, config, run_id, output_path, silent=False):
        super().__init__(config)

        self.run_id = run_id
        self.silent = silent
        self.output_path = output_path

        # Slightly hacky way of allowing for inference-only use
        if run_id is not None:
            os.makedirs(os.path.join(self.output_path, self.config.tag, self.run_id, "model"))
            with open(os.path.join(self.output_path, self.config.tag, self.run_id, "config.json"), "w") as f:
                json.dump(config.data, f, indent=4)

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
        elif self.config.training.opt == 'ranger':
            self.optimizer = Ranger(self.model.parameters())


        else:
            raise Exception("Unrecognised optimiser: " + self.config.training.opt)

    def create_samplers(self):
        self.decode_greedy = GreedySampler(self.config, self.device)
        self.decode_beam = BeamSearchSampler(self.config, self.device)
        self.decode_teacher_force = TeacherForcedSampler(self.config, self.device)
        self.decode_nucleus = ParallelNucleusSampler(self.config, self.device)

        self.reranker = RerankingReducer(self.config, self.device)

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        self.logger.info("Loading from checkpoint " + file_name)
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
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

    def save_checkpoint(self, file_name="checkpoint.pth.tar"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """

        torch.save(
            {
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
                "best_metric": self.best_metric,
            },
            os.path.join(self.output_path, self.config.tag, self.run_id, "model", file_name),
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

    def infer(self, input, reduce_outputs=True):
        """
        Run inference on a dictionary of strings
        """
        batch = self.text_to_batch(input, self.device)

        _, output, output_lens, scores = self.step_validate(
            batch, tgt_field=self.tgt_field, sample_outputs=True, calculate_loss=False, reduce_outputs=reduce_outputs
        )

        if reduce_outputs:
            output_strings = [BPE.decode(output.data[ix][: output_lens[ix]]) for ix in range(len(output_lens))]
        else:
            # There's an extra dim of nesting
            output_strings = [
                [BPE.decode(output.data[i][j][: output_lens[i][j]]) for j in range(len(output_lens[i]))]
                for i in range(len(output_lens))
            ]

        return output_strings, scores

    # def step_train(self, batch):
    #     """
    #     Implement the main logic here for a single training step
    #     """
    #     pass

    # def step_validate(self, batch):
    #     """
    #     Implement the main logic here for a single validation/inference step
    #     """
    #     pass

    def step_train(self, batch, tgt_field):
        loss = 0

        output, logits, _ = self.decode_teacher_force(self.model, batch, tgt_field)

        this_loss = self.loss(logits.permute(0, 2, 1), batch[tgt_field])

        loss += torch.mean(torch.sum(this_loss, dim=1) / (batch[tgt_field + "_len"] - 1).to(this_loss), dim=0)

        return loss

    def train(self):
        """
        Main training loop
        :return:
        """
        if self.tgt_field is None:
            raise Exception("You need to specify the target output field! ie which element of a batch is the output")

        # self.global_step = self.current_epoch * len(self.data_loader.train_loader.dataset)
        # self.global_step = self.current_epoch * len(self.data_loader.train_loader.dataset)//self.config.training.optim_batch_size
        update_mckenzie(0, "-")

        # If we're starting above zero, means we've loaded from chkpt -> validate to give a starting point for fine tuning
        if self.global_step > 0:
            self.begin_epoch_hook()
            self.validate(save=True)

        for epoch in range(self.config.training.num_epochs):
            self.begin_epoch_hook()

            self.train_one_epoch()

            self.current_epoch += 1

            if self.current_epoch > self.config.training.warmup_epochs:
                self.validate(save=True)

            if "bleu" in self.all_metrics_at_best:
                update_mckenzie(
                    (epoch + 1) / self.config.training.num_epochs * 100,
                    "{:0.2f}".format(
                        self.all_metrics_at_best[self.config.training.data.get("mckenzie_metric", "bleu")]
                    ),
                )
            else:
                update_mckenzie((epoch + 1) / self.config.training.num_epochs * 100, "-")

        self.logger.info("## Training completed {:} epochs".format(self.current_epoch + 1))
        self.logger.info("## Best metrics: {:}".format(self.all_metrics_at_best))

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        self.model.train()

        self.logger.info("## Training epoch {:}".format(self.current_epoch))

        self.optimizer.zero_grad()
        steps_accum = 0

        start_step = self.global_step

        for batch_idx, batch in enumerate(
            tqdm(self.data_loader.train_loader, desc="Epoch {:}".format(self.current_epoch), disable=self.silent)
        ):
            batch = {k: (v.to(self.device) if k[-5:] != "_text" else v) for k, v in batch.items()}
            curr_batch_size = batch[[k for k in batch.keys()][0]].size()[0]

            # self.global_step += curr_batch_size

            # Weight the loss by the ratio of this batch to optimiser step size, so that LR is equivalent even when grad accumulation happens
            loss = (
                self.step_train(batch, self.tgt_field)
                * float(curr_batch_size)
                / float(self.config.training.optim_batch_size)
            )

            loss.backward()

            # print(loss)
            # for name, param in self.model.named_parameters():
            #     if "encoder_pooling." in name:
            #     # if "encoder_projection" in name:
            #         print(name, param.requires_grad, param.grad is None)
            #         # print(torch.sum(torch.abs(param.grad)))
            #         if param.requires_grad and param.grad is None:
            #             # print(dir(param))
            #             print(name)

            # exit()

            steps_accum += curr_batch_size

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_gradient)
            
            lr = get_lr(self.config.training.lr, self.global_step, self.config.training.lr_schedule, self.config.training.data.get("lr_warmup", True))

            add_to_log("train/lr", lr, self.global_step, self.config.tag + "/" + self.run_id, self.output_path)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient accumulation
            if steps_accum >= self.config.training.optim_batch_size:
                self.optimizer.step()
                self.optimizer.zero_grad()
                steps_accum = 0
                self.global_step += 1

            if batch_idx % self.config.training.log_interval == 0:
                add_to_log(
                    "train/loss",
                    loss,
                    self.global_step,
                    self.config.tag + "/" + self.run_id,
                    self.output_path
                )

                # TODO: This is currently paraphrase specific! May work for other models but isn't guaranteed
                if batch_idx % (self.config.training.log_interval * 20) == 0 and not self.silent:

                    with torch.no_grad():
                        greedy_output, _, output_lens = self.decode_greedy(self.model, batch, self.tgt_field)

                    if self.tgt_field == "s2":
                        print(BPE.decode(batch[self.src_field][0][: batch[self.src_field + "_len"][0]]))
                    print(BPE.decode(batch[self.tgt_field][0][: batch[self.tgt_field + "_len"][0]]))
                    print(BPE.decode(greedy_output.data[0][: output_lens[0]]))

                torch.cuda.empty_cache()

            if (
                self.config.training.data.get("epoch_steps", 0) > 0
                and self.global_step - start_step >= self.config.training.epoch_steps
            ):
                print(
                    "Epoch step size is set - validatin after {:} training steps".format(
                        self.config.training.epoch_steps
                    )
                )
                break

    def step_validate(self, batch, tgt_field, sample_outputs=True, calculate_loss=True, reduce_outputs=True):

        if not sample_outputs:
            dev_output = None
            dev_output_lens = None
            dev_scores = None
        elif self.config.eval.sampler == "nucleus":
            beam_output, beam_scores, beam_lens = self.decode_nucleus(self.model, batch, tgt_field)

            # dev_output = beam_output[:, 0, :]
            # dev_output_lens = beam_lens[:, 0]
        elif self.config.eval.sampler == "beam":
            beam_output, beam_scores, beam_lens = self.decode_beam(self.model, batch, tgt_field)

            # dev_output = beam_output[:, 0, :]
            # dev_output_lens = beam_lens[:, 0]

        elif self.config.eval.sampler == "greedy":
            greedy_output, greedy_scores, greedy_lens = self.decode_greedy(self.model, batch, tgt_field)
            # Greedy returns a single estimate, so expand to create a fake "beam"
            beam_output = greedy_output.unsqueeze(1)
            beam_scores = greedy_scores.unsqueeze(1)
            beam_lens = greedy_lens.unsqueeze(1)
        else:
            raise Exception("Unknown sampling method!")

        # Rerank (or just select top-1)
        if sample_outputs and reduce_outputs:
            dev_output, dev_output_lens, dev_scores = self.reranker(
                beam_output, beam_lens, batch, tgt_field, scores=beam_scores, presorted=True
            )
        elif sample_outputs:
            dev_output, dev_output_lens, dev_scores = self.reranker(
                beam_output, beam_lens, batch, tgt_field, scores=beam_scores, presorted=True, top1=False
            )

        if calculate_loss:
            _, logits, _ = self.decode_teacher_force(self.model, batch, tgt_field)
            this_loss = self.loss(logits.permute(0, 2, 1), batch[tgt_field])
            normed_loss = torch.mean(torch.sum(this_loss, dim=1) / (batch[tgt_field + "_len"] - 1).to(this_loss), dim=0)
        else:
            normed_loss = None
        return normed_loss, dev_output, dev_output_lens, dev_scores

    def validate(self, save=False, force_save_output=False, use_test=False, use_train=False):
        """
        One cycle of model validation
        :return:
        """

        if self.tgt_field is None:
            raise Exception("You need to specify the target output field! ie which element of a batch is the output")

        self.logger.info("## Validating after {:} epochs".format(self.current_epoch))
        self.model.eval()
        test_loss = 0
        correct = 0

        pred_output = []
        gold_output = []
        gold_input = []  # needed for SARI

        if use_test:
            print("***** USING TEST SET ******")
            valid_loader = self.data_loader.test_loader
        elif use_train:
            print("***** USING TRAINING SET ******")
            valid_loader = self.data_loader.train_loader
        else:
            valid_loader = self.data_loader.valid_loader

        with torch.no_grad():
            num_batches = 0
            for batch_idx, batch in enumerate(
                tqdm(valid_loader, desc="Validating after {:} epochs".format(self.current_epoch), disable=self.silent)
            ):
                batch = {k: (v.to(self.device) if k[-5:] != "_text" else v) for k, v in batch.items()}

                curr_batch_size = batch[[k for k in batch.keys()][0]].size()[0]

                this_loss, dev_output, dev_output_lens, dev_scores = self.step_validate(
                    batch,
                    self.tgt_field,
                    sample_outputs=self.config.eval.data.get("sample_outputs", True),
                    reduce_outputs=(self.config.eval.data.get("topk", 1) == 1),
                )

                test_loss += this_loss

                num_batches += 1

                #  Handle top-1 sampling
                if self.config.eval.data.get("sample_outputs", True) and (self.config.eval.data.get("topk", 1) == 1):
                    for ix, pred in enumerate(dev_output.data):
                        pred_output.append(BPE.decode(pred[: dev_output_lens[ix]]))
                    for ix, gold in enumerate(batch[self.tgt_field]):
                        gold_output.append(BPE.decode(gold[: batch[self.tgt_field + "_len"][ix]]))
                    for ix, gold in enumerate(batch[self.src_field]):
                        gold_input.append(BPE.decode(gold[: batch[self.src_field + "_len"][ix]]))

                    if batch_idx % 200 == 0 and not self.silent:
                        print(gold_output[-2:])
                        print(pred_output[-2:])

                # Handle top-k sampling
                if self.config.eval.data.get("sample_outputs", True) and not (
                    self.config.eval.data.get("topk", 1) == 1
                ):
                    for ix, pred in enumerate(dev_output.data):
                        pred_output.append(
                            [BPE.decode(pred[jx][: dev_output_lens[ix][jx]]) for jx in range(len(pred))]
                        )
                    for ix, gold in enumerate(batch[self.tgt_field]):
                        gold_output.append(BPE.decode(gold[: batch[self.tgt_field + "_len"][ix]]))

                    # if batch_idx > 20:
                    #     break

            test_loss /= num_batches
            self.logger.info("Dev set: Average loss: {:.4f}".format(test_loss))

        add_to_log("dev/loss", test_loss, self.global_step, self.config.tag + "/" + self.run_id, self.output_path)

        if self.config.eval.data.get("sample_outputs", True) and (self.config.eval.data.get("topk", 1) == 1):
            dev_bleu = 100 * bleu_corpus(gold_output, pred_output)

            dev_sari = 100 * np.mean(
                [SARIsent(gold_input[ix], pred_output[ix], [gold_output[ix]]) for ix in range(len(pred_output))]
            )

            dev_em = np.mean([gold_output[ix] == pred_output[ix] for ix in range(len(pred_output))])

            add_to_log("dev/bleu", dev_bleu, self.global_step, self.config.tag + "/" + self.run_id, self.output_path)

            self.logger.info("BLEU: {:}".format(dev_bleu))
            self.logger.info("EM: {:}".format(dev_em))
            self.logger.info("SARI: {:}".format(dev_sari))

        else:
            dev_bleu = "n/a"

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
            with open(os.path.join(self.output_path, self.config.tag, self.run_id, "output.txt"), "w") as f:
                f.write("\n".join([json.dumps(pred) for pred in pred_output]))

            self.all_metrics_at_best = {"nll": test_loss.item()}

            if self.config.eval.data.get("sample_outputs", True) and (self.config.eval.data.get("topk", 1) == 1):
                self.all_metrics_at_best = {
                    **self.all_metrics_at_best,
                    "bleu": dev_bleu,
                    "em": dev_em,
                    "sari": dev_sari,
                }

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

            self.save_checkpoint()

        return test_loss, self.all_metrics_at_best
