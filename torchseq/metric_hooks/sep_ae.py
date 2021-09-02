import jsonlines
import torch
import numpy as np
import sacrebleu
import copy
import os
from abc import abstractmethod
from tqdm import tqdm

from collections import defaultdict
from torchseq.metric_hooks.base import MetricHook
from torchseq.utils.tokenizer import Tokenizer
from torchseq.utils.metrics import bleu_corpus
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config

from torch.autograd import Variable
from torchseq.utils.functions import onehot


import logging


class SepAEMetricHook(MetricHook):

    type = "slow"  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation

    def __init__(self, config, src_field=None, tgt_field=None):
        super().__init__(config, src_field, tgt_field)

        self.logger = logging.getLogger("SepAEMetric")

    def on_begin_epoch(self, use_test=False):
        self.scores = {}

    def on_batch(self, batch, logits, output, memory, use_test=False):
        pass

    def on_end_epoch(self, agent, use_test=False):
        # Temporarily change the config so the bottleneck is noiseless
        var_weight = self.config.bottleneck.data.get("prior_var_weight", 1.0)
        self.config.bottleneck.data["prior_var_weight"] = 0.0

        sample_outputs = self.config.data["eval"].get("sample_outputs", True)
        self.config.eval.data["sample_outputs"] = True

        if (
            self.config.eval.metrics.sep_ae.get("run_codepred", False)
            and self.config.bottleneck.get("code_predictor", None) is not None
        ):
            self.logger.info("Running generation with code prediction")
            (
                self.scores["sepae_codepred_bleu"],
                self.scores["sepae_codepred_selfbleu"],
                self.scores["sepae_codepred_ibleu"],
            ), _ = SepAEMetricHook.eval_gen_codepred_v2(
                self.config,
                agent,
                test=use_test,
                train_code_predictor=self.config.eval.metrics.sep_ae.get("train_codepred", True),
                cache_data=self.config.eval.metrics.sep_ae.get("cache_data", True),
                single_training_target=self.config.eval.metrics.sep_ae.get("single_target", False),
                enforce_unique_codes=self.config.eval.metrics.sep_ae.get("enforce_unique_codes", False),
            )
            self.logger.info("...done")

        if self.config.eval.metrics.sep_ae.get("run_oracle", True):
            self.logger.info("Running generation with oracle template")
            self.scores["sepae_oracle"] = SepAEMetricHook.eval_gen_with_oracle(self.config, agent, test=use_test)
            self.logger.info("...done")

        if self.config.eval.metrics.sep_ae.get("run_noised", False):
            self.logger.info("Running generation with noised encodings")
            (
                self.scores["cluster_gen_noised_diversity_bleu"],
                self.scores["cluster_gen_noised_diversity_selfbleu"],
                self.scores["cluster_gen_noised_diversity_ibleu"],
            ) = SepAEMetricHook.eval_gen_noised_diversity(self.config, agent)
            self.logger.info("...done")

        # self.logger.info("Running generation to test reconstruction")
        # self.scores["sepae_reconstruction"] = SepAEMetricHook.eval_reconstruction(self.config, agent)
        # self.logger.info("...done")

        # Reset the config
        self.config.bottleneck.data["prior_var_weight"] = var_weight
        self.config.eval.data["sample_outputs"] = sample_outputs
        return self.scores

    @abstractmethod
    def eval_gen_with_oracle(config, agent, test=False, use_qqp=False):
        config_gen_with_templ = copy.deepcopy(config.data)
        config_gen_with_templ["dataset"] = "json"
        config_gen_with_templ["json_dataset"] = {
            "path": config.eval.metrics.sep_ae.eval_dataset,
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "syn_input", "to": "template"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }
        config_gen_with_templ["eval"]["topk"] = 1

        config.bottleneck.data["prior_var_weight"] = 0.0

        data_loader = JsonDataLoader(config=Config(config_gen_with_templ))

        _, _, (output, _, _), memory = agent.inference(
            data_loader.test_loader if test else data_loader.valid_loader, memory_keys_to_return=["vq_codes"]
        )

        # # Now run again with forced codes
        # Confirmed - this doesn't need to be run separately
        # samples = (data_loader._test if test else data_loader._valid).samples
        # samples = [{**x, "forced_codes": codes} for x, codes in zip(samples, memory["vq_codes"])]
        # # print('From run 1')
        # # print(output[0])
        # # print(samples[0])
        # # print(len(samples), len(memory['vq_codes']))
        # config_gen_with_templ["json_dataset"] = {
        #     "path": None,
        #     "field_map": [
        #         {"type": "copy", "from": "tgt", "to": "s2"},
        #         {"type": "copy", "from": "syn_input", "to": "template"},
        #         {"type": "copy", "from": "sem_input", "to": "s1"},
        #         {"type": "copy", "from": "forced_codes", "to": "forced_codes"},
        #     ],
        # }

        # forced_loader = JsonDataLoader(config=Config(config_gen_with_templ), dev_samples=samples)
        # _, _, (output, _, _), memory_forced = agent.inference(
        #     forced_loader.valid_loader, memory_keys_to_return=["vq_codes"]
        # )

        # print('From run 2')
        # print(output[0])
        # print(memory_forced['vq_codes'][0])

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(
                config.env.data_path,
                config.eval.metrics.sep_ae.eval_dataset,
                f"{split}.jsonl",
            )
        ) as f:
            rows = [row for row in f][: config_gen_with_templ["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows]
        inputs = [[q["sem_input"]] for q in rows]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu = sacrebleu.corpus_bleu(output, list(zip(*inputs))).score

        alpha = config.eval.metrics.sep_ae.get("ibleu_alpha", 0.8)
        ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

        return (tgt_bleu, self_bleu, ibleu)

    @abstractmethod
    def eval_gen_noised_diversity(config, agent, noise_weight=2.0, code_offset=2, test=False):
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": "wikianswers-para-splitforgeneval",
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }
        config_gen_noised["eval"]["topk"] = 1
        config_gen_noised["eval"]["repeat_samples"] = 5

        if config_gen_noised["bottleneck"]["vector_quantized"]:
            var_offset = config_gen_noised["bottleneck"]["quantizer_num_residual"]
            print(config_gen_noised["bottleneck"]["quantizer_heads"], var_offset, code_offset)
            agent.model.bottleneck.quantizer._code_offset = (
                [0] * var_offset
                + [code_offset]
                + [code_offset] * (config_gen_noised["bottleneck"]["quantizer_heads"] - var_offset - 1)
            )
            print(agent.model.bottleneck.quantizer._code_offset)
        else:
            var_offset = config_gen_noised["bottleneck"]["num_similar_heads"]
            if config_gen_noised["bottleneck"].get("invert_templ", False):
                var1 = noise_weight
                var2 = 0.0
            else:
                var1 = 0.0
                var2 = noise_weight
            config.bottleneck.data["prior_var_weight"] = (
                [var1] * var_offset + [var2] + [var2] * (config_gen_noised["encdec"]["num_heads"] - var_offset - 1)
            )

        data_loader = JsonDataLoader(config=Config(config_gen_noised))

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(config.env.data_path, f"wikianswers-para-splitforgeneval/{split}.jsonl")
        ) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]
        refs = [q["paras"] for q in rows for _ in range(5)]
        inputs = [[q["sem_input"]] for q in rows for _ in range(5)]

        # refs = [x["paras"] for x in qs_by_para_split]
        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        config.bottleneck.data["prior_var_weight"] = 0.0
        agent.model.bottleneck.quantizer._code_offset = 0

        tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu = sacrebleu.corpus_bleu(output, list(zip(*inputs))).score

        alpha = config.eval.metrics.sep_ae.get("ibleu_alpha", 0.8)
        ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

        return (tgt_bleu, self_bleu, ibleu)

    @abstractmethod
    def eval_gen_random(config, agent, test=False):
        # Now run eval
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": config.eval.metrics.sep_ae.eval_dataset,
            "field_map": [
                {"type": "copy", "from": "sem_input", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "template"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
                {"type": "copy", "from": "forced_codes", "to": "forced_codes"},
            ],
        }
        config_gen_noised["eval"]["topk"] = 1

        config.bottleneck.data["prior_var_weight"] = 0.0

        split = "test" if test else "dev"
        with jsonlines.open(
            os.path.join(
                config.env.data_path,
                config.eval.metrics.sep_ae.eval_dataset,
                f"{split}.jsonl",
            )
        ) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]

        num_heads = config.bottleneck.code_predictor.num_heads
        codebook_size = config.bottleneck.codebook_size

        rows = [{**row, "forced_codes": np.random.randint(0, codebook_size - 1, num_heads).tolist()} for row in rows]

        data_loader = JsonDataLoader(config=Config(config_gen_noised), test_samples=rows)

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader)

        refs = [q["paras"] for q in rows]
        inputs = [[q["sem_input"]] for q in rows]

        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        # config.bottleneck.data["prior_var_weight"] = 0.0

        tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu = sacrebleu.corpus_bleu(output, list(zip(*inputs))).score

        alpha = config.eval.metrics.sep_ae.get("ibleu_alpha", 0.8)
        ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

        return (tgt_bleu, self_bleu, ibleu), output

    @abstractmethod
    def eval_gen_codepred_v2(
        config,
        agent,
        test=False,
        use_qqp=False,
        train_code_predictor=True,
        cache_data=False,
        single_training_target=False,
        enforce_unique_codes=False,
    ):
        logger = logging.getLogger("SepAEMetric")
        sample_outputs = config.data["eval"].get("sample_outputs", True)

        if train_code_predictor:
            # Generate the training data
            # TODO: move these to the config
            # lr = 1e-4
            bsz = config.bottleneck.code_predictor.bsz
            num_steps = config.bottleneck.code_predictor.num_steps
            MAX_SAMPLES = config.bottleneck.code_predictor.max_samples

            logger.info("Generating encodings and vq codes to train code predictor")

            dataset_all = config.eval.metrics.sep_ae.flattened_dataset
            dataset_clusters = config.eval.metrics.sep_ae.cluster_dataset

            # if use_qqp:
            #     dataset_all = "qqp-allqs"
            #     dataset_clusters = "qqp-clusters"
            #     # dataset_geneval = "qqp-splitforgeneval"
            # else:
            #     dataset_all = "wikianswers-para-allqs"
            #     dataset_clusters = "wikianswers-pp"
            #     # dataset_geneval = "wikianswers-para-splitforgeneval"

            if os.path.exists(agent.run_output_path + "/codepred_cache_X.npy"):
                print("Loading from cache")
                X = np.load(agent.run_output_path + "/codepred_cache_X.npy")
                y = np.load(agent.run_output_path + "/codepred_cache_y.npy")
                X_dev = np.load(agent.run_output_path + "/codepred_cache_X_dev.npy")
                y_dev = np.load(agent.run_output_path + "/codepred_cache_y_dev.npy")
            else:
                print("Cache not found, generating...")
                cfg_dict = copy.deepcopy(config.data)

                config.eval.data["sample_outputs"] = False

                cfg_dict["training"]["batch_size"] = 24
                cfg_dict["eval"]["eval_batch_size"] = 24
                cfg_dict["training"]["dataset"] = "json"
                cfg_dict["eval"]["truncate_dataset"] = MAX_SAMPLES
                cfg_dict["training"]["shuffle_data"] = False
                cfg_dict["json_dataset"] = {
                    "path": dataset_all,
                    "field_map": [
                        {"type": "copy", "from": "q", "to": "s2"},
                        {"type": "copy", "from": "q", "to": "s1"},
                    ],
                }
                cfg_dict["bottleneck"]["prior_var_weight"] = 0.0

                data_loader = JsonDataLoader(Config(cfg_dict))

                _, _, _, memory_train = agent.inference(
                    data_loader.train_loader, memory_keys_to_return=["sep_encoding_1", "sep_encoding_2", "vq_codes"]
                )

                X = np.concatenate(
                    [memory_train["sep_encoding_1"][:, 0, :], memory_train["sep_encoding_2"][:, 0, :]], axis=1
                )
                y = memory_train["vq_codes"][:, :, 0].tolist()

                del memory_train

                _, _, _, memory_dev = agent.inference(
                    data_loader.valid_loader, memory_keys_to_return=["sep_encoding_1", "sep_encoding_2", "vq_codes"]
                )

                X_dev = np.concatenate(
                    [memory_dev["sep_encoding_1"][:, 0, :], memory_dev["sep_encoding_2"][:, 0, :]], axis=1
                )
                y_dev = memory_dev["vq_codes"][:, :, 0].tolist()

                del memory_dev

                if cache_data:
                    np.save(agent.run_output_path + "/codepred_cache_X.npy", X)
                    np.save(agent.run_output_path + "/codepred_cache_y.npy", y)
                    np.save(agent.run_output_path + "/codepred_cache_X_dev.npy", X_dev)
                    np.save(agent.run_output_path + "/codepred_cache_y_dev.npy", y_dev)

                print("Cache built")

            with jsonlines.open(os.path.join(config.env.data_path, dataset_clusters, "train.jsonl")) as f:
                train_qs = [row for row in f]
            train_cluster_ixs = []
            ix = 0
            for cix, cluster in enumerate(train_qs):
                clen = len(cluster["qs"])
                if ix + clen > MAX_SAMPLES:
                    break
                for i in range(clen):
                    cluster_ixs = list(range(ix, ix + clen))
                    # if args.dataset != 'qqp':
                    cluster_ixs.remove(ix + i)
                    if enforce_unique_codes:
                        cluster_ixs = [cix for cix in cluster_ixs if (y[cix] != y[ix]).any()]

                    train_cluster_ixs.append(cluster_ixs)
                ix += clen

            with jsonlines.open(os.path.join(config.env.data_path, dataset_clusters, "dev.jsonl")) as f:
                dev_qs = [row for row in f]
            dev_cluster_ixs = []
            ix = 0
            for cix, cluster in enumerate(dev_qs[:MAX_SAMPLES]):
                clen = len(cluster["qs"])
                if ix + clen > MAX_SAMPLES:
                    break
                for i in range(clen):
                    cluster_ixs = list(range(ix, ix + clen))
                    # if args.dataset != 'qqp':
                    cluster_ixs.remove(ix + i)
                    dev_cluster_ixs.append(cluster_ixs)
                ix += clen
            # Train the code predictor

            logger.info("Training code predictor")
            logger.info("Single training: {:}".format("on" if single_training_target else "off"))

            rand_ixs = np.random.randint(0, high=len(train_cluster_ixs), size=(num_steps, bsz))

            best_dev_loss = 1e10

            # Train from scratch

            for layer in agent.model.code_predictor.classifier.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

            for iter in tqdm(range(num_steps), desc="Training code predictor", disable=agent.silent):
                batch_ixs = rand_ixs[iter, :]

                inputs = Variable(torch.tensor([X[ix] for ix in batch_ixs if len(train_cluster_ixs[ix]) > 0])).cuda()

                if config.bottleneck.get("quantizer_transitions", False) or single_training_target:
                    tgt_ixs = [
                        np.random.choice(train_cluster_ixs[cix])
                        for cix in batch_ixs
                        if len(train_cluster_ixs[cix]) > 0
                    ]
                    tgt = torch.cat(
                        [
                            onehot(torch.tensor(y[tgt_ix]), N=config.bottleneck.code_predictor.output_dim).unsqueeze(0)
                            * 1.0
                            for tgt_ix in tgt_ixs
                        ],
                        dim=0,
                    ).cuda()
                else:
                    tgt_ixs = [
                        [y[ix] for ix in train_cluster_ixs[cix]][:100]
                        for cix in batch_ixs
                        if len(train_cluster_ixs[cix]) > 0
                    ]
                    max_len = max([len(tgt) for tgt in tgt_ixs])
                    # pad
                    tgt_ixs = torch.LongTensor([tgt + [tgt[0]] * (max_len - len(tgt)) for tgt in tgt_ixs]).cuda()
                    tgt = onehot(tgt_ixs, N=config.bottleneck.code_predictor.output_dim).sum(dim=1)
                    tgt = torch.where(tgt > 0, 1, 0)
                    # tgt = torch.where(torch.cat([torch.sum(torch.cat([onehot(torch.tensor(y[ix]), N=config.bottleneck.code_predictor.output_dim).unsqueeze(0) for ix in train_cluster_ixs[cix][:20]], dim=0), dim=0, keepdims=True) for cix in batch_ixs], dim=0) > 0, 1, 0).cuda()

                train_loss = agent.model.code_predictor.train_step(inputs, tgt)

                if iter % 1000 == 0:
                    agent.model.code_predictor.eval()

                    dev_loss = 0

                    for x_ix, cluster in enumerate(dev_cluster_ixs):
                        if len(cluster) == 0:
                            continue
                        inputs = Variable(torch.tensor([X_dev[x_ix]])).cuda()

                        if config.bottleneck.get("quantizer_transitions", False) or single_training_target:
                            tgt_ixs = [np.random.choice(dev_cluster_ixs[cix]) for cix in [x_ix]]
                            dev_tgt = torch.cat(
                                [
                                    onehot(
                                        torch.tensor(y_dev[tgt_ix]), N=config.bottleneck.code_predictor.output_dim
                                    ).unsqueeze(0)
                                    * 1.0
                                    for tgt_ix in tgt_ixs
                                ],
                                dim=0,
                            ).cuda()
                        else:
                            tgt_ixs = [[y_dev[ix] for ix in cluster][:100]]
                            max_len = max([len(tgt) for tgt in tgt_ixs])
                            # pad
                            tgt_ixs = torch.LongTensor(
                                [tgt + [tgt[0]] * (max_len - len(tgt)) for tgt in tgt_ixs]
                            ).cuda()
                            tgt = onehot(tgt_ixs, N=config.bottleneck.code_predictor.output_dim).sum(dim=1)
                            dev_tgt = torch.where(tgt > 0, 1, 0)
                            # dev_tgt = torch.where(torch.cat([torch.sum(torch.cat([onehot(torch.tensor(y[ix]), N=config.bottleneck.code_predictor.output_dim).unsqueeze(0) for ix in cluster[:20]], dim=0), dim=0, keepdims=True)], dim=0) > 0, 1, 0).cuda()

                        dev_loss += agent.model.code_predictor.train_step(inputs, dev_tgt, take_step=False)
                        # outputs = agent.model.code_predictor(inputs)

                        # logits = [outputs[:, 0, :].unsqueeze(1)]

                        # for head_ix in range(1, config.bottleneck.code_predictor.num_heads):
                        #     if config.bottleneck.get("quantizer_transitions", False):
                        #         prev_oh = (
                        #             onehot(
                        #                 torch.max(logits[-1], dim=-1).indices,
                        #                 N=config.bottleneck.code_predictor.output_dim,
                        #             )
                        #             * 1.0
                        #         )
                        #         logits.append(
                        #             outputs[:, head_ix, :].unsqueeze(1)
                        #             + agent.model.code_predictor.transitions[head_ix - 1](prev_oh)
                        #         )
                        #     else:
                        #         logits.append(outputs[:, head_ix, :].unsqueeze(1))
                        # logits = torch.cat(logits, dim=1)

                        # dev_loss += (
                        #     torch.sum(
                        #         -1
                        #         * torch.nn.functional.log_softmax(logits, dim=-1)
                        #         * dev_tgt
                        #         / dev_tgt.sum(dim=-1, keepdims=True),
                        #         dim=-1,
                        #     )
                        #     .mean()
                        #     .detach()
                        # )

                    dev_loss /= x_ix

                    if dev_loss < best_dev_loss:
                        logger.info("Saving...")
                        agent.save_checkpoint()
                        best_dev_loss = dev_loss
                    logger.info("Iteration: {}. Loss: {}. Train loss {}.".format(iter, dev_loss.item(), train_loss))

            # Now reload the checkpoint - this emulates early stopping, for the code predictor
            save_path = os.path.join(agent.run_output_path, "model", "checkpoint.pt")
            agent.load_checkpoint(save_path)

        # Now run eval
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": config.eval.metrics.sep_ae.eval_dataset,
            "field_map": [
                {"type": "copy", "from": "sem_input", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "template"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }
        config_gen_noised["eval"]["topk"] = 1

        config.bottleneck.code_predictor.data["infer_codes"] = True

        config.bottleneck.data["prior_var_weight"] = 0.0

        data_loader = JsonDataLoader(config=Config(config_gen_noised))

        agent.config.eval.data["sample_outputs"] = True

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        config.eval.data["sample_outputs"] = sample_outputs

        config.bottleneck.code_predictor.data["infer_codes"] = False  # reset this flag

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(
                config.env.data_path,
                config.eval.metrics.sep_ae.eval_dataset,
                f"{split}.jsonl",
            )
        ) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]

        refs = [q["paras"] for q in rows]
        inputs = [[q["sem_input"]] for q in rows]

        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        # config.bottleneck.data["prior_var_weight"] = 0.0

        tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu = sacrebleu.corpus_bleu(output, list(zip(*inputs))).score

        alpha = config.eval.metrics.sep_ae.get("ibleu_alpha", 0.8)
        ibleu = alpha * tgt_bleu - (1 - alpha) * self_bleu

        return (tgt_bleu, self_bleu, ibleu), output

    @abstractmethod
    def eval_gen_beam_diversity(config, agent, test=False):
        config_gen_noised = copy.deepcopy(config.data)
        config_gen_noised["dataset"] = "json"
        config_gen_noised["json_dataset"] = {
            "path": "wikianswers-para-splitforgeneval",
            "field_map": [
                {"type": "copy", "from": "tgt", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }

        prev_topk = config.eval.get("topk", 1)
        config.eval.data["topk"] = 4

        data_loader = JsonDataLoader(config=Config(config_gen_noised))

        _, _, (output, _, _), _ = agent.inference(data_loader.test_loader if test else data_loader.valid_loader)

        NUM_HYPOTHESES = 4

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(config.env.data_path, f"wikianswers-para-splitforgeneval/{split}.jsonl")
        ) as f:
            rows = [row for row in f][: config_gen_noised["eval"].get("truncate_dataset", None)]

        output = [q for beam in output for q in beam]

        # Now, score the multiple outputs within each example
        refs = [q["paras"] for q in rows for _ in range(NUM_HYPOTHESES)]
        N = NUM_HYPOTHESES
        # Get the other outputs from the same grouping
        other_outs = [
            [q for j, q in enumerate(output[N * (i // N) : N * (i // N + 1)]) if j % N != i % N]
            for i in range(len(output))
        ]

        max_num_refs = max([len(x) for x in refs])
        refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

        alpha = config.eval.metrics.sep_ae.get("ibleu_alpha", 0.8)
        tgt_bleu_div = sacrebleu.corpus_bleu(output, list(zip(*refs_padded))).score
        self_bleu_div = sacrebleu.corpus_bleu(output, list(zip(*other_outs))).score
        ibleu_div = alpha * tgt_bleu_div - (1 - alpha) * self_bleu_div

        config.eval.data["topk"] = prev_topk

        return (tgt_bleu_div, self_bleu_div, ibleu_div)
