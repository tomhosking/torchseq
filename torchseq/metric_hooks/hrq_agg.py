import jsonlines
import torch
import numpy as np
import copy
import os
import json
import re
from abc import abstractmethod
from tqdm import tqdm
from math import ceil


from collections import defaultdict, Counter
from torchseq.metric_hooks.base import MetricHook
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config
from torchseq.utils.functions import batchify, cos_sim
from torchseq.utils.rouge import get_jackknife_rouge, get_pairwise_rouge

import sacrebleu
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk import sent_tokenize, word_tokenize

from openTSNE import TSNE

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import logging
from truecase import get_true_case

logger = logging.getLogger("HRQAggregationMetric")


# Check for equivalence of paths, allowing for wildcard values
def paths_are_equal(p1, p2):
    if len(p1) != len(p2):
        return False
    for x1, x2 in zip(p1, p2):
        if x1 != x2 and x1 != "_" and x2 != "_":
            return False
    return True


class AggregationTree:
    def __init__(self, elements):
        self.nodes = defaultdict(float)
        for element in elements:
            self.nodes[element] += 1

    def __getitem__(self, key):
        val = self.nodes[key]
        for k, v in self.nodes.items():
            if len(k) > len(key) and k[: len(key)] == key:
                val += v
        return val

    def __setitem__(self, key, val):
        self.nodes[key] = val

    def pop(self, key):
        self.nodes.pop(key)

    def __len__(self):
        return len(self.nodes)

    def items(self):
        return self.nodes.items()

    def keys(self):
        return self.nodes.keys()

    def values(self):
        return self.nodes.values()


class HRQAggregationMetricHook(MetricHook):
    type = "slow"  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation

    # def __init__(self, config, src_field=None, tgt_field=None):
    #     super().__init__(config, src_field, tgt_field)

    def on_begin_epoch(self, use_test=False):
        self.scores = {}

    def on_batch(self, batch, logits, output, memory, use_test=False):
        pass

    def on_end_epoch(self, agent, use_test=False):
        # Populate caches
        logger.info("Populating HRQ caches - this may take a while!")
        _, _ = HRQAggregationMetricHook.codes_from_cache(self.config, agent, test=False, train=False)
        # _, _ = HRQAggregationMetricHook.codes_from_cache(self.config, agent, test=False, train=True)
        logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_nli", False):
            logger.info("Running NLI eval")
            self.scores["hrq_agg_nli"], _, _ = HRQAggregationMetricHook.eval_nli(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_tsne", False):
            logger.info("Running tsne eval")
            _ = HRQAggregationMetricHook.eval_tsne(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_specialisation", False):
            logger.info("Running specialisation eval")
            self.scores["hrq_agg_specialisation"] = HRQAggregationMetricHook.eval_specialisation(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_generate_summaries", False):
            logger.info("Running generation using HRQ paths")
            (
                self.scores["hrq_agg_generation"],
                generated_summaries,
            ) = HRQAggregationMetricHook.eval_generate_summaries_and_score(
                self.config,
                agent,
                test=use_test,
            )
            split = "test" if use_test else "dev"
            with open(agent.run_output_path + f"/summaries_{split}.json", "w") as f:
                json.dump(generated_summaries, f)
            logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_masked_generation", False):
            logger.info("Running generation with masking")
            self.scores["hrq_agg_masking"], masked_outputs = HRQAggregationMetricHook.eval_masked_generation(
                self.config,
                agent,
                test=use_test,
            )
            split = "test" if use_test else "dev"
            with open(agent.run_output_path + f"/masked_outputs_{split}.json", "w") as f:
                json.dump(masked_outputs, f)
            logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_oracle", False):
            logger.info("Running oracle summary generation")
            res = HRQAggregationMetricHook.eval_oracle_summaries(self.config, agent, test=use_test)
            self.scores["hrq_agg_oracle"] = {"full": res[0], "masked": res[1]}
            split = "test" if use_test else "dev"
            with open(agent.run_output_path + f"/oracle_eval_{split}.json", "w") as f:
                json.dump(res, f)
            logger.info("...done")

        # if self.config.eval.metrics.hrq_agg.get("run_purity", False):
        #     logger.info("Running cluster purity eval")
        #     self.scores["hrq_agg_purity"] = HRQAggregationMetricHook.eval_cluster_purity(
        #         self.config, agent, test=use_test
        #     )
        #     logger.info("...done")

        return self.scores

    @abstractmethod
    def codes_from_cache(
        config,
        agent,
        test=False,
        train=False,
        eval=False,
        force_rebuild=False,
        sample_outputs=True,
        save_to_cache=True,
    ):
        if eval:
            raise Exception("codes_from_cache not yet implemented for eval datasets!")
        split = "test" if test else ("train" if train else "dev")
        cache_key = split + ("_eval" if eval else "")
        if (
            not force_rebuild
            and agent.run_output_path is not None
            and os.path.exists(agent.run_output_path + f"/sents_by_code_{cache_key}.json")
        ):
            with open(agent.run_output_path + f"/sents_by_code_{cache_key}.json") as f:
                sents_by_code = json.load(f)
            with open(agent.run_output_path + f"/outputs_with_codes_{cache_key}.json") as f:
                outputs_with_codes = json.load(f)
        else:
            cfg_dict = copy.deepcopy(config.data)

            dataset = config.eval.metrics.hrq_agg.dataset_eval if eval else config.eval.metrics.hrq_agg.dataset_all

            cfg_dict["json_dataset"] = {
                "path": dataset,
                "filename": "{split}" if eval else "reviews.{split}",
                "field_map": [
                    {"type": "copy", "from": "sentence", "to": "target"},
                    {"type": "copy", "from": "sentence", "to": "source"},
                ],
            }

            # cfg_dict["eval"]["metrics"]["hrq_agg"] = {
            #     "dataset_clusters": "opagg/space-filtered-25toks-clusters",
            #     "dataset_all": "opagg/space-filtered-25toks-all",
            #     "run_generate_summaries": False,
            #     "run_retrieval": False,
            #     "run_masked_generation": True,
            # }

            cfg_dict["training"]["batch_size"] = cfg_dict["eval"]["eval_batch_size"]
            cfg_dict["training"]["shuffle_data"] = False
            # cfg_dict['eval']['truncate_dataset'] = 10000

            config_forced = Config(cfg_dict)

            # checkpoint_path = path_to_model + "/model/checkpoint.pt"
            # instance = Seq2SeqAgent(config=config, run_id=None, output_path=None, data_path='../../data/', silent=False, verbose=False, training_mode=False)
            # instance.load_checkpoint(checkpoint_path)
            # instance.model.eval()

            data_loader = JsonDataLoader(config_forced, data_path=agent.data_path)

            loader = (
                data_loader.test_loader if test else (data_loader.train_loader if train else data_loader.valid_loader)
            )

            sample_outputs_prev_state = agent.config.eval.get("sample_outputs", True)
            agent.config.eval.data["sample_outputs"] = sample_outputs

            _, _, (pred_output, _, _), memory = agent.inference(
                loader, memory_keys_to_return=["vq_codes"], desc="Calculating encodings"
            )

            agent.config.eval.data["sample_outputs"] = sample_outputs_prev_state

            with jsonlines.open(agent.data_path + "/" + dataset + f"/reviews.{split}.jsonl") as f:
                inputs = [x["sentence"] for x in f]
            # with jsonlines.open(agent.data_path + "/opagg/space-filtered-all/reviews.dev.jsonl") as f:
            #     scores = [x["rating"] for x in f]

            sents_by_code = defaultdict(set)
            for sentence, codes in zip(inputs, memory["vq_codes"]):
                sents_by_code[str(tuple(codes.tolist()))].add(sentence)

            for k, v in sents_by_code.items():
                sents_by_code[k] = list(v)

            outputs_with_codes = {
                "outputs": pred_output if sample_outputs else None,
                "codes": memory["vq_codes"].tolist(),
                "inputs": inputs,
            }

            if save_to_cache:
                with open(agent.run_output_path + f"/sents_by_code_{cache_key}.json", "w") as f:
                    json.dump(sents_by_code, f)
                with open(agent.run_output_path + f"/outputs_with_codes_{cache_key}.json", "w") as f:
                    json.dump(outputs_with_codes, f)

        return sents_by_code, outputs_with_codes

    @abstractmethod
    def eval_nli(config, agent, test=False, show_plot=False):
        split = "test" if test else "dev"

        cfg_dict = copy.deepcopy(config.data)

        # cfg_dict = config_nli.data

        cfg_dict["json_dataset"] = {
            "path": config.eval.metrics.hrq_agg.dataset_all,
            "filename": "reviews.{split}",
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "target"},
                {"type": "copy", "from": "sentence", "to": "source"},
            ],
        }
        # cfg_dict["eval"]["eval_batch_size"] = 32
        cfg_dict["eval"]["truncate_dataset"] = 1000

        config_nli = Config(cfg_dict)

        data_loader = JsonDataLoader(config_nli, data_path=agent.data_path)

        # checkpoint_path = path_to_model + "/model/checkpoint.pt"
        # instance = Seq2SeqAgent(config=config, run_id=None, output_path=None, data_path='../../data/', silent=True, verbose=False, training_mode=False)
        # instance.load_checkpoint(checkpoint_path)
        # instance.model.eval()

        config_forced = copy.deepcopy(config_nli.data)
        config_forced["dataset"] = "json"
        config_forced["json_dataset"] = {
            "path": config.eval.metrics.hrq_agg.dataset_all,
            "filename": "reviews.{split}",
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "target"},
                {"type": "copy", "from": "sentence", "to": "source"},
                {"type": "copy", "from": "head_mask", "to": "head_mask"},
            ],
        }

        num_heads = config_nli.bottleneck.modules[0].quantizer.num_heads

        samples = data_loader._valid.samples

        output_masked = {}
        probs_masked = {}
        preds_by_depth = {}

        inputs = [x["sentence"] for x in samples]

        # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base-mnli")
        tokenizer = AutoTokenizer.from_pretrained("tomhosking/deberta-v3-base-debiased-nli")

        # model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base-mnli").cuda()
        model = AutoModelForSequenceClassification.from_pretrained("tomhosking/deberta-v3-base-debiased-nli").cuda()

        # from sentence_transformers import SentenceTransformer

        # model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

        # inputs_embedded = model.encode(inputs)

        print("HRQ fwd pass")
        mean_lens = {}
        for mask_len in tqdm(range(0, num_heads)):
            mask = [1] * (num_heads - mask_len) + [0] * mask_len
            masked_samples = [{**x, "head_mask": mask} for x in samples]

            forced_loader = JsonDataLoader(
                config=Config(config_forced), dev_samples=masked_samples, data_path=agent.data_path
            )

            _, _, (output, _, _), _ = agent.inference(
                forced_loader.valid_loader, desc=f"Decoding with mask len {mask_len}"
            )

            output_masked[mask_len] = output
            preds_by_depth[mask_len] = []

            mean_lens[mask_len] = np.mean([len(x.split()) for x in output])

            torch.cuda.empty_cache()

            #     print('Paraphrase detection')
            #     outputs_embedded = model.encode(output)
            #     sims = np.diagonal(outputs_embedded.dot(inputs_embedded.T)/(np.linalg.norm(outputs_embedded, axis=-1)*np.linalg.norm(inputs_embedded, axis=-1)))
            #     preds_by_depth[mask_len] = sims
            #     probs_masked[mask_len] = np.mean(sims)

        print("Batched NLI")
        ENTAILMENT_LABEL = (
            model.config.label2id["ENTAILMENT"]
            if "ENTAILMENT" in model.config.label2id
            else model.config.label2id["entailment"]
        )
        for tgt_mask_len in tqdm(range(0, num_heads)):
            probs_masked[tgt_mask_len] = {}
            preds_by_depth[tgt_mask_len] = {}

            preds_by_depth[tgt_mask_len]["inputs"] = []
            for _, batch in batchify(list(zip(inputs, output_masked[tgt_mask_len])), 16):
                hf_inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")

                outputs = model(**hf_inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)
                preds_by_depth[tgt_mask_len]["inputs"].extend(pred.tolist())

                torch.cuda.empty_cache()

            probs_masked[tgt_mask_len]["inputs"] = np.mean(
                [(x == ENTAILMENT_LABEL) * 1.0 for x in preds_by_depth[tgt_mask_len]["inputs"]], axis=0
            )

            for src_mask_len in range(0, num_heads):
                preds_by_depth[tgt_mask_len][src_mask_len] = []
                for _, batch in batchify(list(zip(output_masked[src_mask_len], output_masked[tgt_mask_len])), 16):
                    hf_inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")

                    outputs = model(**hf_inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    pred = torch.argmax(probs, dim=-1)
                    preds_by_depth[tgt_mask_len][src_mask_len].extend(pred.tolist())

                    torch.cuda.empty_cache()

                probs_masked[tgt_mask_len][src_mask_len] = np.mean(
                    [(x == ENTAILMENT_LABEL) * 1.0 for x in preds_by_depth[tgt_mask_len][src_mask_len]], axis=0
                )

        y_labels = ["inputs"] + list(range(num_heads))
        x_labels = list(range(num_heads))

        im = [[probs_masked[x][y] if x != y else 0 for x in x_labels] for y in y_labels]

        plt.figure()
        ax = plt.gca()

        plt.imshow(im)
        ax.set_title(agent.run_id)
        ax.set_xlabel("Hypothesis (mask length)")
        ax.set_ylabel("Premise (mask length)")
        ax.set_xticks([i for i in range(num_heads)])
        ax.set_yticks([i for i in range(num_heads + 1)])
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        plt.colorbar()

        plt.savefig(agent.run_output_path + "/entailment_grid.pdf", bbox_inches="tight")

        if show_plot:
            plt.show()

        with open(agent.run_output_path + f"/output_masked_{split}.json", "w") as f:
            json.dump(output_masked, f)

        # Calculate mean entailment scores
        scores = {
            "fwd_mean": np.mean(
                [p for i, row in probs_masked.items() for j, p in row.items() if j != "inputs" and j < i]
            ),
            "bwd_mean": np.mean(
                [p for i, row in probs_masked.items() for j, p in row.items() if j != "inputs" and j > i]
            ),
            "inputs": np.mean([row["inputs"] for i, row in probs_masked.items()]),
        }
        scores["diff"] = scores["fwd_mean"] - scores["bwd_mean"]

        return scores, output_masked, preds_by_depth

    @abstractmethod
    def eval_tsne(config, agent, test=False, show_plot=False):
        logger.info("Loading data")
        sents_by_code, outputs_with_codes = HRQAggregationMetricHook.codes_from_cache(config, agent, test)

        split = "test" if test else "dev"

        LIMIT = 5000
        PLOT_LIMIT = 250

        sent_codes = [tuple(x) for x in outputs_with_codes["codes"]][:LIMIT]
        outputs = outputs_with_codes["outputs"][:LIMIT]

        num_heads = config.bottleneck.modules[0].quantizer.num_heads
        embedding_dim = config.bottleneck.embedding_dim

        embeddings = agent.model.bottleneck.module_list[0].quantizer._embedding

        def get_embedding(path, depth=None):
            if depth is None:
                depth = len(path)
            embedded = (
                torch.cat(
                    [
                        embeddings[hix](torch.LongTensor([path[hix]]).to(agent.device))
                        for hix in range(len(path[:depth]))
                    ],
                    dim=0,
                )
                .sum(dim=0)
                .detach()
                .cpu()
            )

            return embedded

        # all_codes_d3 = [(i,j,0) for i in range(12) for j in range(12)] # for k in range(12)
        all_codes_d3 = list(set([x[:4] for x in sent_codes]))
        # logger.info(len(all_codes_d3), " will be plotted")

        # codes = all_codes_d3
        codes = None

        embedded_codes = (
            torch.cat(
                [
                    torch.cat(
                        [
                            embeddings[hix](torch.LongTensor([x[hix]]).to(agent.device))
                            for hix in range(min(3, num_heads))
                        ],
                        dim=0,
                    ).unsqueeze(0)
                    for x in all_codes_d3
                ],
                dim=0,
            )
            .detach()
            .cpu()
        )
        # partial_codes = torch.cat(
        #     [torch.sum(embedded_codes[:, :(hix), :], dim=1, keepdim=True) for hix in range(3 + 1)], dim=1
        # )
        full_embeddings = torch.sum(embedded_codes, dim=1)

        with jsonlines.open(
            agent.data_path + "/" + config.eval.metrics.hrq_agg.dataset_all + f"/reviews.{split}.jsonl"
        ) as f:
            entity_ids = [x["entity_id"] for x in f][:LIMIT]
        with jsonlines.open(
            agent.data_path + "/" + config.eval.metrics.hrq_agg.dataset_all + f"/reviews.{split}.jsonl"
        ) as f:
            review_ids = [x["review_id"] for x in f][:LIMIT]

        # construct probabilistic paths

        paths_by_entity = defaultdict(lambda: defaultdict(int))
        paths_by_entity_probs = defaultdict(lambda: defaultdict(float))

        for entity_id, review_id, path, pred_sent in zip(entity_ids, review_ids, sent_codes, outputs):
            for h in range(4):
                paths_by_entity[entity_id][path[: h + 1]] += 1

        # normalise
        for entity_id in paths_by_entity.keys():
            for h in range(4):
                max_paths = 4 ** (h + 1)
                total = sum(
                    sorted([v for k, v in paths_by_entity[entity_id].items() if len(k) == (h + 1)], reverse=True)[
                        :max_paths
                    ]
                )

                for k, v in sorted(
                    [(k, v) for k, v in paths_by_entity[entity_id].items() if len(k) == (h + 1)],
                    key=lambda x: x[1],
                    reverse=True,
                )[:max_paths]:
                    if h == 0 or k[:-1] in paths_by_entity_probs[entity_id]:
                        paths_by_entity_probs[entity_id][k] = 1.0 * v / total

        tsne = TSNE(n_components=2)

        logger.info("Fitting tSNE")

        X_full_embedded = tsne.fit(full_embeddings)

        logger.info("Transforming paths")

        # X_byhead_embedded = X_full_embedded.transform(partial_codes.reshape(-1, 768)).reshape(-1, num_heads + 1, 2)

        # colors = list(mcolors.XKCD_COLORS.values())
        # colors = list(mcolors.CSS4_COLORS.values())
        # np.random.shuffle(colors)
        colors = list(mcolors.TABLEAU_COLORS.values())  # + list(mcolors.BASE_COLORS.values())
        np.random.shuffle(colors)

        color_labels = [colors[x[0] % len(colors)] for x in all_codes_d3]

        marker_types = [
            "o",
            "p",
            "h",
            "H",
            "*",
            "P",
            "s",
            "p",
            "X",
            "|",
            "_",
            "D",
            "v",
            "^",
            "<",
            ">",
            "8",
            "s",
            "p",
            "1",
            "2",
            "3",
            "4",
            ".",
            "d",
            ",",
            "o",
        ]
        pattern_types = ("-", "+", "x", "\\", "*", "o", "O", ".", "/")

        linecols = [
            "tab:orange",
            "tab:blue",
            "tab:green",
            "tab:red",
            "red",
            "blue",
            "orange",
            "grey",
            "grey",
            "grey",
            "grey",
        ]

        logger.info("Plotting points")

        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # unique_entities = list(set(entity_ids))

        if len(all_codes_d3[0]) > 1:
            markers = [marker_types[x[1] % len(marker_types)] for x in all_codes_d3]
        else:
            markers = [marker_types[0] for x in all_codes_d3]

        if len(all_codes_d3[0]) > 2 and False:
            patterns = [pattern_types[x[2] % len(pattern_types)] for x in all_codes_d3]
        else:
            patterns = [pattern_types[0] for x in all_codes_d3]

        for i, (x, y, c, m, p) in enumerate(
            zip(X_full_embedded.T[0], X_full_embedded.T[1], color_labels, markers, patterns)
        ):
            if i > PLOT_LIMIT:
                break
            ax.scatter(x, y, color=c, s=20, marker=m)  # hatch=4*p, facecolor='white'

        # entitycols = [linecols[unique_entities.index(x) % len(linecols)] for x in entity_ids]

        # plt.title(agent.run_id)

        logger.info("Plotting lines")
        # plotted = set()
        # if num_heads > 1:
        #     entities_plotted = {}
        #     for i in range(LIMIT):
        #         # if unique_entities.index(entity_ids[i]) > 1:
        #         #     continue
        #         if len(entities_plotted) == 2 and entity_ids[i] not in entities_plotted:
        #             continue
        #         entities_plotted[entity_ids[i]] = len(entities_plotted)
        #         for hix in range(max(min(num_heads - 1, 2), 0)):
        #             if (entity_ids[i], codes[i][: hix + 1]) in plotted:
        #                 continue
        #             from_coords = X_byhead_embedded[i : (i + 1), hix, :]
        #             to_coords = X_byhead_embedded[i : (i + 1), hix + 1, :]
        #             ab_pairs = np.c_[from_coords, to_coords]
        #             ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)
        #             alpha = np.sqrt(paths_by_entity_probs[entity_ids[i]][codes[i][: hix + 1]])
        #             plotted.add((entity_ids[i], codes[i][: hix + 1]))
        #             ax.plot(*ab_args, c=linecols[entities_plotted[entity_ids[i]]], linewidth=5, alpha=alpha)

        ent_count = 0

        entities_to_plot = list(paths_by_entity_probs.keys())[:2]

        for entity in entities_to_plot:
            tree = paths_by_entity_probs[entity]
            the_other_entity = entities_to_plot[-1 - 1 * ent_count]
            if ent_count == 2:
                break
            ent_count += 1

            for depth in range(1, 4):
                for codes, weight in tree.items():
                    if len(codes) == depth:
                        # plot it!
                        if len(codes) > 1:
                            embedding_from = get_embedding(codes, depth - 1)
                        else:
                            embedding_from = np.zeros(embedding_dim)
                        from_coords = X_full_embedded.transform(embedding_from.reshape(-1, embedding_dim)).reshape(
                            -1, 2
                        )
                        embedding_to = get_embedding(codes, depth)
                        to_coords = X_full_embedded.transform(embedding_to.reshape(-1, embedding_dim)).reshape(-1, 2)

                        ab_pairs = np.c_[from_coords, to_coords]
                        ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)  # + (0.2) * ent_count

                        #                 print(codes, weight, ab_args)
                        if weight < 0.1 ** (depth + 1):
                            continue
                        #                 alpha = np.sqrt(weight) + 0.3*depth + 0.1
                        alpha = min(max(np.cbrt(weight), 0.2) + 0.1 * (max(2 - depth, 0)), 1.0)
                        color = (
                            "tab:brown"
                            if (
                                codes in paths_by_entity_probs[the_other_entity]
                                and paths_by_entity_probs[the_other_entity][codes] > 0.1 ** (depth + 1)
                            )
                            else linecols[ent_count - 1]
                        )
                        ax.plot(*ab_args, c=color, linewidth=12 / (1.5 + depth), alpha=alpha)  #
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(agent.run_output_path + "/tsne_entity_overlay.pdf", bbox_inches="tight")
        # plt.savefig("../../plots/acl23/tsne_entity_overlay.pdf", bbox_inches="tight")

        if show_plot:
            plt.show()

        return {"probability_tree": paths_by_entity_probs}

    @abstractmethod
    def get_feature_entropies(codes, features, num_heads):
        entropies_by_depth = {}
        for d in range(num_heads):
            features_by_code = defaultdict(Counter)
            for feature, code in zip(features, codes):
                features_by_code[code[: (d + 1)]][feature] += 1

            entropies = []
            for code in features_by_code.keys():
                denom = sum([x for x in features_by_code[code].values()])
                distribution = [1.0 * x / denom for x in features_by_code[code].values()]

                this_entropy = np.sum(-1.0 * np.log(distribution) * distribution)
                entropies.append(this_entropy)
            entropies_by_depth[d] = np.mean(entropies)

        return entropies_by_depth

    @abstractmethod
    def eval_specialisation(config, agent, test=False):
        sents_by_code, outputs_with_codes = HRQAggregationMetricHook.codes_from_cache(config, agent, test)
        split = "test" if test else "dev"

        codes = [tuple(x) for x in outputs_with_codes["codes"]]

        num_heads = config.bottleneck.modules[0].quantizer.num_heads
        codebook_size = config.bottleneck.modules[0].quantizer.codebook_size

        with jsonlines.open(
            agent.data_path + "/" + config.eval.metrics.hrq_agg.dataset_all + f"/reviews.{split}.jsonl"
        ) as f:
            entity_ids = [x["entity_id"] for x in f]
        with jsonlines.open(
            agent.data_path + "/" + config.eval.metrics.hrq_agg.dataset_all + f"/reviews.{split}.jsonl"
        ) as f:
            review_ids = [x["review_id"] for x in f]
        with jsonlines.open(
            agent.data_path + "/" + config.eval.metrics.hrq_agg.dataset_all + f"/reviews.{split}.jsonl"
        ) as f:
            inputs = [x["sentence"] for x in f]
        with jsonlines.open(
            agent.data_path + "/" + config.eval.metrics.hrq_agg.dataset_all + f"/reviews.{split}.jsonl"
        ) as f:
            ratings = [x["rating"] for x in f]

        # Get (weak) aspect labels
        space_aspect_list = ["building", "cleanliness", "food", "location", "rooms", "service"]
        aspect_keywords = defaultdict(list)
        for aspect in space_aspect_list:
            with open(agent.data_path + f"/opagg/aspect-seeds/{aspect}.txt") as f:
                keywords = [line.strip().split()[1] for line in f.readlines()]
            aspect_keywords[aspect] = keywords
        keywords_to_aspect = {kw: aspect for aspect, kws in aspect_keywords.items() for kw in kws}
        aspect_labels = []
        for sentence in inputs:
            labels = set()
            for kw, aspect in keywords_to_aspect.items():
                if kw in sentence.split():
                    labels.add(aspect)
            if len(labels) == 0:
                labels.add("UNK")
            aspect_labels.append(labels)

        sentence_positions = [i - review_ids.index(review_id) for i, review_id in enumerate(review_ids)]
        review_lengths = [review_ids.count(review_id) for i, review_id in enumerate(review_ids)]
        sentence_pos_relative = [1.0 * pos / length for pos, length in zip(sentence_positions, review_lengths)]
        sentence_pos_buckets = [0 if pos < 0.25 else (2 if pos >= 0.75 else 1) for pos in sentence_pos_relative]

        # Measures of path concentration

        # entropy (path | entity_id)

        # construct probabilistic paths
        paths_by_entity = defaultdict(lambda: defaultdict(int))
        paths_by_review = defaultdict(lambda: defaultdict(int))
        paths_by_entity_probs = defaultdict(lambda: defaultdict(float))
        paths_by_review_probs = defaultdict(lambda: defaultdict(float))

        for entity_id, review_id, path in zip(entity_ids, review_ids, codes):
            for h in range(num_heads):
                paths_by_entity[entity_id][path[: h + 1]] += 1
                paths_by_review[review_id][path[: h + 1]] += 1

        # normalise
        entropies_entity_by_head = {}
        entropies_review_by_head = {}

        for h in range(num_heads):
            entropies_entity_by_head[h] = []
            entropies_review_by_head[h] = []

            for entity_id in paths_by_entity.keys():
                total = sum([v for k, v in paths_by_entity[entity_id].items() if len(k) == (h + 1)])
                for k, v in paths_by_entity[entity_id].items():
                    if len(k) == (h + 1):
                        paths_by_entity_probs[entity_id][k] = 1.0 * v / total
                this_probs = [prob for k, prob in paths_by_entity_probs[entity_id].items() if len(k) == (h + 1)]
                this_entropy = np.sum(-1.0 * np.log(this_probs) * this_probs)
                entropies_entity_by_head[h].append(this_entropy)

            entropies_entity_by_head[h] = np.mean(entropies_entity_by_head[h])

            for review_id in paths_by_review.keys():
                total = sum([v for k, v in paths_by_review[review_id].items() if len(k) == (h + 1)])
                for k, v in paths_by_review[review_id].items():
                    if len(k) == (h + 1):
                        paths_by_review_probs[review_id][k] = 1.0 * v / total
                this_probs = [prob for k, prob in paths_by_review_probs[review_id].items() if len(k) == (h + 1)]
                this_entropy = np.sum(-1.0 * np.log(this_probs) * this_probs)
                entropies_review_by_head[h].append(this_entropy)

            entropies_review_by_head[h] = np.mean(entropies_review_by_head[h])

        uniform_by_depth = {}
        for d in range(num_heads):
            codebook_size = codebook_size if isinstance(codebook_size, int) else codebook_size[d]
            uniform_prob = 1.0 / (codebook_size ** (d + 1))
            uniform_entropy = -1.0 * (codebook_size ** (d + 1)) * (np.log(uniform_prob) * uniform_prob)
            uniform_by_depth[d] = uniform_entropy

        # Measures of predictability

        # entropy (aspect | cluster)
        # entropy (rating | cluster)
        # entropy (sent position in review | cluster)

        scores = {
            "entropy_uniform_codes": uniform_by_depth,
            "entropy_codes_from_entities": entropies_entity_by_head,
            "entropy_codes_from_reviews": entropies_review_by_head,
            "entropy_aspects_from_codes": HRQAggregationMetricHook.get_feature_entropies(
                codes,
                [list(x)[0] for x in aspect_labels],
                num_heads,
            ),
            "entropy_ratings_from_codes": HRQAggregationMetricHook.get_feature_entropies(
                codes,
                ratings,
                num_heads,
            ),
            "entropy_positions_from_codes": HRQAggregationMetricHook.get_feature_entropies(
                codes,
                sentence_pos_buckets,
                num_heads,
            ),
        }

        return scores

    @abstractmethod
    def select_entity_summary_paths(
        all_review_paths,
        max_path_depth,
        path_limit=5,
        truncation_length=None,
        prune_min_weight=None,
        prune_max_paths=None,
        use_tfidf=False,
        tfidf_weighting_scheme=2,
        block_paths=None,
        term_score_weights=None,
        allow_wildcards=False,
        silent=True,
    ):
        summary_paths = {}
        summary_path_weights = {}

        # remove blocked paths

        if block_paths is not None:
            all_review_paths_filtered = copy.deepcopy(all_review_paths)
            for entity_id, reviews in all_review_paths.items():
                for rev_id, paths in reviews.items():
                    for path in paths:
                        for blocked in block_paths[entity_id]:
                            if (
                                tuple(path[: len(blocked)]) == blocked
                                and path in all_review_paths_filtered[entity_id][rev_id]
                            ):
                                all_review_paths_filtered[entity_id][rev_id] = [
                                    x for x in all_review_paths_filtered[entity_id][rev_id] if x != path
                                ]
            all_review_paths = all_review_paths_filtered

        if use_tfidf:
            terms_by_doc = defaultdict(Counter)
            # all_terms = set()
            idf = {}

            for entity_id, reviews in all_review_paths.items():
                all_paths = [tuple(path) for paths in reviews.values() for path in paths]
                for path in all_paths:
                    for max_len in range(1, min(max_path_depth + 1, len(path) + 1)):
                        term = path[:max_len]
                        terms_by_doc[entity_id][term] += 1
                        if allow_wildcards:
                            for wild_ix in range(max_len - 1):  # don't add wildcards to the end
                                wildcard_term = term[:wild_ix] + tuple(["_"]) + term[(wild_ix + 1) :]
                                terms_by_doc[entity_id][wildcard_term] += 1

                # Starting with the most specific terms, check whether their ancestors have no other descendants (ie, prefer the more specific term)
                for term, count in sorted(terms_by_doc[entity_id].items(), key=lambda x: len(x[0]), reverse=True):
                    for max_len in range(1, len(term)):
                        if terms_by_doc[entity_id][term[:-max_len]] == count:
                            # Zero out the generic term in favour of the specific one (since it )
                            terms_by_doc[entity_id][term[:-max_len]] = 0

            all_terms = set([term for terms in terms_by_doc.values() for term in terms.keys()])

            for term in all_terms:
                idf[term] = np.log(
                    len(all_review_paths) / sum([1 for doc_terms in terms_by_doc.values() if term in doc_terms])
                )

            for entity_id, reviews in tqdm(all_review_paths.items(), desc="Calculating term scores", disable=silent):
                path_weights = {}
                for term in terms_by_doc[entity_id].keys():
                    tf = terms_by_doc[entity_id][term] / sum(terms_by_doc[entity_id].values())

                    if term_score_weights is not None:
                        # Multiply by the term weights, with backoff
                        for path_len in range(len(term), 0, -1):
                            if term[:path_len] in term_score_weights:
                                tf *= term_score_weights[term[:path_len]]
                                break

                    if tfidf_weighting_scheme == 1:
                        path_weights[term] = tf * np.sqrt(idf[term]) / len(term)
                    elif tfidf_weighting_scheme == 2:
                        path_weights[term] = tf * np.sqrt(idf[term] / len(term))
                    elif tfidf_weighting_scheme == 3:
                        path_weights[term] = tf * (idf[term]) / len(term)
                    elif tfidf_weighting_scheme == 4:
                        path_weights[term] = tf * len(term)
                    elif tfidf_weighting_scheme == 5:
                        path_weights[term] = tf * np.sqrt(len(term))
                    elif tfidf_weighting_scheme == 6:
                        path_weights[term] = tf
                    elif tfidf_weighting_scheme == 7:
                        path_weights[term] = tf * (idf[term]) / np.sqrt(len(term))
                    elif tfidf_weighting_scheme == 8:
                        path_weights[term] = tf * (idf[term])
                    elif tfidf_weighting_scheme == 9:
                        path_weights[term] = tf * np.sqrt(idf[term])
                    elif tfidf_weighting_scheme == 10:
                        path_weights[term] = np.sqrt(tf) * idf[term]
                    # path_weights[term] = tf * 3**(len(term)-1)

                summary_paths[entity_id], summary_path_weights[entity_id] = [], []
                for path, score in sorted(path_weights.items(), key=lambda x: x[1], reverse=True):
                    if len(summary_paths[entity_id]) >= path_limit:
                        break
                    if (
                        # len(
                        #     [
                        #         1
                        #         for length in range(len(path))
                        #         if path[:length] in summary_paths[entity_id]
                        #     ]
                        # )
                        # == 0
                        # and
                        len(
                            [
                                1
                                for selected_path in summary_paths[entity_id]
                                # if path[: min(len(selected_path), len(path))]
                                # == selected_path[: min(len(selected_path), len(path))]
                                if paths_are_equal(
                                    path[: min(len(selected_path), len(path))],
                                    selected_path[: min(len(selected_path), len(path))],
                                )
                                or (
                                    len(path) > 1
                                    and len(selected_path) > 1
                                    # and path[: min(len(selected_path), len(path)) - 1]
                                    # == selected_path[: min(len(selected_path), len(path)) - 1]
                                    and paths_are_equal(
                                        path[: min(len(selected_path), len(path)) - 1],
                                        selected_path[: min(len(selected_path), len(path)) - 1],
                                    )
                                )
                            ]
                        )
                        == 0
                    ):
                        summary_paths[entity_id].append(path)
                        summary_path_weights[entity_id].append(score)

            return summary_paths, summary_path_weights

        for entity_id, reviews in all_review_paths.items():
            all_paths = [tuple(path) for paths in reviews.values() for path in paths]

            path_weights = AggregationTree(all_paths)

            if truncation_length is not None:
                path_weights_pruned = Counter()
                for codes, weight in path_weights.items():
                    path_weights_pruned[codes[:truncation_length]] += weight
                summary_paths[entity_id] = [x[0] for x in path_weights_pruned.most_common(path_limit)]
                summary_path_weights[entity_id] = [x[1] for x in path_weights_pruned.most_common(path_limit)]
            elif prune_max_paths is not None:
                while len(path_weights) > prune_max_paths:
                    remaining_paths = [x for x in path_weights.items()]  # if len(x[0]) > 1
                    if len(remaining_paths) == 0:
                        break
                    code_to_prune, weight = sorted(remaining_paths, key=lambda x: (x[1], -len(x[0])))[
                        0
                    ]  # sort lowest weight, then longest path
                    path_weights.nodes[code_to_prune] = 0
                    if len(code_to_prune) > 1:
                        path_weights.nodes[code_to_prune[:-1]] += weight
                    path_weights.nodes.pop(code_to_prune)
                summary_paths[entity_id] = list(path_weights.keys())[:path_limit]
                summary_path_weights[entity_id] = list(path_weights.values())[:path_limit]
            elif prune_min_weight is not None:
                total = sum(path_weights.nodes.values())
                for k, v in path_weights.nodes.items():
                    path_weights.nodes[k] = v / total
                if len(path_weights) == 0:
                    print("Agg tree has no contents!")
                    print(all_paths)
                    print(path_weights)
                while (min(path_weights.values())) < prune_min_weight and len(path_weights) > path_limit:
                    remaining_paths = [x for x in path_weights.items()]  # if len(x[0]) > 1
                    if len(remaining_paths) == 0:
                        break
                    code_to_prune, weight = sorted(remaining_paths, key=lambda x: (x[1], -len(x[0])))[
                        0
                    ]  # sort lowest weight, then longest path
                    path_weights.nodes[code_to_prune] = 0
                    if len(code_to_prune) > 1:
                        path_weights.nodes[code_to_prune[:-1]] += weight
                    path_weights.nodes.pop(code_to_prune)
                    if len(code_to_prune) == 0:
                        # rebalance
                        total = sum(path_weights.nodes.values())
                        for k, v in path_weights.nodes.items():
                            path_weights.nodes[k] = v / total
                summary_paths[entity_id], summary_path_weights[entity_id] = zip(
                    *sorted(path_weights.items(), key=lambda x: x[1], reverse=True)[:path_limit]
                )
                summary_paths[entity_id], summary_path_weights[entity_id] = list(summary_paths[entity_id]), list(
                    summary_path_weights[entity_id]
                )
                # summary_paths[entity_id] = list(path_weights.keys())[:path_limit]
                # summary_path_weights[entity_id] = list(path_weights.values())[:path_limit]

            else:
                raise "You need to specify at least one parameter to select_entity_summary_paths!"

        return summary_paths, summary_path_weights

    """Generate summaries from sets of input reviews, by aggregating their hierarchical encodings.
    Parameters:
    agent       -- A trained summarisation model
    test        -- Use test split?
    eval_data   -- [{'entity_id': 0, 'reviews': [{'review_id': 12345, 'sentences': ['Review sentences go here']}]}
                    or set to None to load from config.eval.metrics.hrq_agg.dataset_eval
    """

    @abstractmethod
    def eval_generate_summaries_and_score(config, agent, test=False, eval_data=None, term_score_weights=None):
        split = "test" if test else "dev"

        if eval_data is None:
            dataset_eval = config.eval.metrics.hrq_agg.get("dataset_eval", "opagg/space-eval")
            with jsonlines.open(os.path.join(agent.data_path, dataset_eval, f"{split}.jsonl")) as reader:
                eval_data = [x for x in reader]

        # First, get encodings for all sentences
        config_codes = copy.deepcopy(config.data)
        config_codes["dataset"] = "json"
        config_codes["json_dataset"] = {
            "path": None,
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "target"},
                {"type": "copy", "from": "sentence", "to": "source"},
            ],
        }

        # MASK_LENGTH = 4

        if agent.config.bottleneck.get("quantizer_heads", None) is not None:
            num_heads = agent.config.bottleneck.quantizer_heads - agent.config.bottleneck.get(
                "quantizer_num_residual", 0
            )
        else:
            bneck_types = [x.type for x in agent.config.bottleneck.modules]
            if "hrqvae" not in bneck_types:
                logger.warning("Tried to run hrq aggregation on a model without HRQ!")
                return {}, []
            quantizer_index = bneck_types.index("hrqvae")
            num_heads = agent.config.bottleneck.modules[quantizer_index].quantizer.num_heads

        space_aspect_list = ["building", "cleanliness", "food", "location", "rooms", "service"]
        aspect_keywords = defaultdict(list)
        for aspect in space_aspect_list:
            with open(agent.data_path + f"/opagg/aspect-seeds/{aspect}.txt") as f:
                keywords = [line.strip().split()[1] for line in f.readlines()]
            aspect_keywords[aspect] = keywords

        all_aspect_keywords = [kw for kws in aspect_keywords.values() for kw in kws]

        all_aspect_keywords += [
            "good",
            "bad",
            "ok",
            "great",
            "poor",
            "fine",
            "excellent",
            "terrible",
            "awful",
            "disappointing",
            "amazing",
            "special",
            "fantastic",
            "wonderful",
        ]
        all_aspect_keywords += [
            "rooms",
            "bed",
            "beds",
            "cookie",
            "cookies",
            "cheap",
            "expensive",
            "positive",
            "negative",
            "quick",
            "slow",
            "fast",
            "better",
            "worse",
            "worn",
            "new",
            "modern",
            "lovely",
            "wifi",
            "recommend",
            "restaurant",
            "restaurants",
            "shuttle",
            "airport",
            "parking",
            "light",
            "dark",
            "luxurious",
            "luxury",
            "price",
            "priced",
            "overpriced",
            "tired",
            "huge",
            "tiny",
        ]

        def prefilter_condition(sentence, hotel_aspect_filter=True, amazon_filter=False, min_length=0, max_length=25):
            if len(sentence.split()) > max_length:
                return False
            if len(sentence.split()) < min_length:
                return False
            if sum(i.isalpha() for i in sentence) / len(sentence) < 0.5:
                return False
            if re.search(r"[0-9]+ night", sentence.lower()) is not None:
                return False
            if (
                " stayed" in sentence.lower() or " visited" in sentence.lower()
            ):  # or " my wife and" in sentence.lower() or " my husband and" in sentence.lower()
                return False
            if amazon_filter and (sentence.lower()[:7] == "this is" or sentence.lower()[:6] == "i love"):
                return False
            if "i bought this" in sentence.lower():
                return False
            # if sentence.lower().count(' we ') > 1 or sentence.lower().count(' i ') > 1:
            #     return False
            overlap = set(word_tokenize(sentence.replace("\n", " ").lower())) & set(all_aspect_keywords)
            if len(overlap) == 0 and hotel_aspect_filter:
                return False
            return True

        # from allennlp.predictors.predictor import Predictor
        # import allennlp_models.structured_prediction
        # from allennlp.models.archival import load_archive

        # predictor = Predictor.from_archive(
        #     load_archive(
        #         "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
        #         cuda_device=torch.cuda.current_device(),
        #     ),
        # )

        eval_sentences = [
            {"sentence": sentence, "review_id": review["review_id"], "entity_id": row["entity_id"]}
            for row in eval_data
            for rev_id, review in enumerate(row["reviews"])
            for sentence in (
                review["sentences"][:-1]
                if config.eval.metrics.hrq_agg.get("summary_skip_last", False)
                else review["sentences"]
            )
            if prefilter_condition(
                sentence,
                hotel_aspect_filter=config.eval.metrics.hrq_agg.get("summary_hotel_aspect_filter", True),
                amazon_filter=config.eval.metrics.hrq_agg.get("summary_amazon_filter", False),
                max_length=config.eval.metrics.hrq_agg.get("summary_max_sentence_length", 25),
                min_length=config.eval.metrics.hrq_agg.get("summary_min_sentence_length", 0),
            )
            and (
                sentence not in [sent for rev in row["reviews"][:rev_id] for sent in rev["sentences"]]
                or not config.eval.metrics.hrq_agg.get("summary_dedupe_sentences", False)
            )
            and len(review["sentences"]) >= config.eval.metrics.hrq_agg.get("summary_min_review_sentences", 0)
            and len(review["sentences"]) <= config.eval.metrics.hrq_agg.get("summary_max_review_sentences", 1000)
        ]

        # eval_sentences = []
        # for i, batch_inputs in tqdm(batchify(eval_sentences_orig, batch_size=24), desc='Splitting eval'):

        #     parses = predictor.predict_batch_json([{'sentence': sent['sentence']} for sent in batch_inputs])
        #     for parse, sent in zip(parses, batch_inputs):
        #         subsents = [node['word'] for node in parse['hierplane_tree']['root']['children'] if node['nodeType'] == 'S']
        #         if len(subsents) > 0:
        #             eval_sentences.extend([{**sent, 'sentence': subsent} for subsent in subsents])
        #         else:
        #             eval_sentences.append(sent)
        # for row in tqdm(eval_sentences_orig):
        #     parse = predictor.predict(row['sentence'])
        #     subsents = [node['word'] for node in parse['hierplane_tree']['root']['children'] if node['nodeType'] == 'S']
        #     if len(subsents) > 0:
        #         for subsent in subsents:
        #             eval_sentences.append({**row, 'sentence': subsent})
        #     else:
        #         eval_sentences.append(row)

        # eval_sentences = [row for row in eval_sentences if prefilter_condition(row['sentence'])]

        data_loader = JsonDataLoader(
            config=Config(config_codes), data_path=agent.data_path, dev_samples=eval_sentences
        )

        sample_outputs = agent.config.eval.get("sample_outputs", True)
        agent.config.eval.data["sample_outputs"] = False

        _, _, _, memory = agent.inference(
            data_loader.valid_loader, memory_keys_to_return=["vq_codes"], desc="Calculating encodings"
        )

        all_codes = memory["vq_codes"].tolist()

        # _, outputs_with_codes = HRQAggregationMetricHook.codes_from_cache(config, agent, eval=True)
        # all_codes = outputs_with_codes['codes']

        # Organise paths by entity/review, then get summary paths
        paths_by_review_by_entity = defaultdict(lambda: defaultdict(list))
        for row, codes in zip(eval_sentences, all_codes):
            paths_by_review_by_entity[row["entity_id"]][row["review_id"]].append(codes)

        if config.eval.metrics.hrq_agg.get("summary_smart_heuristic", False):
            # Get some generic, some specific
            logger.info("Selecting specific terms...")
            (
                summary_paths_specific,
                summary_path_weights_specific,
            ) = HRQAggregationMetricHook.select_entity_summary_paths(
                paths_by_review_by_entity,
                # ceil(num_heads // 3),
                8,
                path_limit=config.eval.metrics.hrq_agg.get("summary_smart_num_specific", 4),
                truncation_length=None,
                prune_min_weight=None,
                prune_max_paths=None,
                use_tfidf=True,
                tfidf_weighting_scheme=config.eval.metrics.hrq_agg.get("summary_smart_specific_weight_scheme", 2),
                silent=agent.silent,
                allow_wildcards=config.eval.metrics.hrq_agg.get("summary_allow_wildcards", False),
                term_score_weights=term_score_weights,
            )
            logger.info("Selecting generic terms...")
            summary_paths_generic, summary_path_weights_generic = HRQAggregationMetricHook.select_entity_summary_paths(
                paths_by_review_by_entity,
                # ceil(num_heads // 2),
                8,
                path_limit=config.eval.metrics.hrq_agg.get("summary_smart_num_generic", 4),
                truncation_length=None,
                prune_min_weight=0.01,
                prune_max_paths=None,
                use_tfidf=True
                if config.eval.metrics.hrq_agg.get("summary_smart_generic_weight_scheme", None) is not None
                else False,
                tfidf_weighting_scheme=config.eval.metrics.hrq_agg.get("summary_smart_generic_weight_scheme", 5),
                # block_paths={k: [p[:1] for p in v] for k, v in summary_paths_specific.items()},
                block_paths={k: v for k, v in summary_paths_specific.items()},
                allow_wildcards=config.eval.metrics.hrq_agg.get("summary_allow_wildcards", False),
                silent=agent.silent,
                term_score_weights=term_score_weights,
            )
            # summary_paths_generic, summary_path_weights_generic = HRQAggregationMetricHook.select_entity_summary_paths(
            #     paths_by_review_by_entity,
            #     ceil(num_heads // 8),
            #     path_limit=config.eval.metrics.hrq_agg.get("summary_smart_num_generic", 4),
            #     truncation_length=None,
            #     prune_min_weight=None,
            #     prune_max_paths=None,
            #     use_tfidf=True,
            #     block_paths={k: [p[:1] for p in v] for k, v in summary_paths_specific.items()},
            # )

            summary_paths, summary_path_weights = {}, {}
            for ent_id in summary_paths_generic.keys():
                summary_paths[ent_id] = summary_paths_generic[ent_id] + summary_paths_specific[ent_id]
                summary_path_weights[ent_id] = (
                    summary_path_weights_generic[ent_id] + summary_path_weights_specific[ent_id]
                )
        else:
            logger.info("Selecting summary terms...")
            summary_paths, summary_path_weights = HRQAggregationMetricHook.select_entity_summary_paths(
                paths_by_review_by_entity,
                # ceil(num_heads // 4),
                8,
                path_limit=config.eval.metrics.hrq_agg.get("summary_path_limit", 6),
                truncation_length=config.eval.metrics.hrq_agg.get("summary_truncation_length", None),
                prune_min_weight=config.eval.metrics.hrq_agg.get("summary_prune_min_weight", 0.01),
                prune_max_paths=config.eval.metrics.hrq_agg.get("summary_prune_max", None),
                use_tfidf=config.eval.metrics.hrq_agg.get("summary_use_tfidf", False),
                tfidf_weighting_scheme=config.eval.metrics.hrq_agg.get("summary_tfidf_weight_scheme", 5),
                silent=agent.silent,
                term_score_weights=term_score_weights,
            )

        # # Identify top clusters per entity
        # codes_by_entity = defaultdict(Counter)

        # for row, codes in zip(eval_data, all_codes):
        #     codes_by_entity[row["entity_id"]][tuple(codes.tolist()[:-MASK_LENGTH])] += 1

        # mask = [1] * (num_heads - MASK_LENGTH) + [0] * MASK_LENGTH

        # filtered_examples = []
        # for entity, counter in codes_by_entity.items():
        #     for x, count in counter.most_common(5):
        #         filtered_examples.append(
        #             {"entity_id": entity, "codes": list(x) + [0] * MASK_LENGTH, "sentence": "", "head_mask": mask}
        #         )

        filtered_examples = []
        # for entity, paths in summary_paths.items():
        for row in eval_data:
            entity_id = row["entity_id"]
            for path in summary_paths[entity_id]:
                mask = [1] * len(path) + [0] * (num_heads - len(path))
                # print(path)
                if "_" in path:
                    path = list(path)
                    mask[path.index("_")] = 0
                    path[path.index("_")] = 0
                filtered_examples.append(
                    {
                        "entity_id": entity_id,
                        "codes": list(path) + [0] * (num_heads - len(path)),
                        "sentence": "",
                        "head_mask": mask,
                    }
                )

        # Generate!
        agent.config.eval.data["sample_outputs"] = True

        config_forced = copy.deepcopy(config.data)
        config_forced["dataset"] = "json"
        config_forced["json_dataset"] = {
            "path": None,
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "target"},
                {"type": "copy", "from": "sentence", "to": "source"},
                {"type": "copy", "from": "codes", "to": "forced_codes"},
                {"type": "copy", "from": "head_mask", "to": "head_mask"},
            ],
        }

        forced_loader = JsonDataLoader(
            config=Config(config_forced), data_path=agent.data_path, dev_samples=filtered_examples
        )

        _, _, (output, _, _), _ = agent.inference(forced_loader.valid_loader, desc="Generating")

        sentences_by_entity = defaultdict(list)
        for input, sentence in zip(filtered_examples, output):
            sentences_by_entity[input["entity_id"]].append(get_true_case(sentence))

        # if config.eval.metrics.hrq_agg.get("summary_dedupe_output", False):
        #     for ent_id, sents in sentences_by_entity.items():
        #         sentences_by_entity[ent_id] = list(set(sents))

        # Extractive summaries
        sentences_by_path = defaultdict(lambda: defaultdict(list))
        extractive_summs = []
        for row, codes in zip(eval_sentences, all_codes):
            sentences_by_path[row["entity_id"]][tuple(codes)].append(row["sentence"])

        all_evidence = []
        all_evidence_paths = []
        for row in tqdm(eval_data, desc="Selecting ROUGE centroids for extractive summary", disable=agent.silent):
            summary_sentences = []
            summary_evidence = []
            summary_evidence_paths = []
            entity = row["entity_id"]
            paths = summary_paths[entity]
            for i, path in enumerate(paths):
                path_evidence = []
                path_evidence_paths = []
                for codes, sents in sentences_by_path[entity].items():
                    # if path == codes[: len(path)]:
                    if paths_are_equal(path, codes[: len(path)]):
                        path_evidence.extend(sents)
                        path_evidence_paths.extend([codes for _ in sents])

                    # Select rouge centroid as top extractive pick

                if len(path_evidence) == 0:
                    print("Found entity with no evidence!")
                    print(entity)
                    print(path_evidence)
                    print(path)
                    print(paths)
                    print(summary_path_weights[entity])
                    print(sentences_by_path[entity].keys())
                    print(paths_by_review_by_entity[entity])

                path_evidence = path_evidence[:50]
                path_evidence_paths = path_evidence_paths[:50]

                scores = []
                for x in path_evidence:
                    scores.append([])
                    for y in path_evidence:
                        rouge = get_pairwise_rouge(x, y)["rouge2"]
                        scores[-1].append(rouge)
                    # print(scores)
                    scores[-1] = np.mean(scores[-1])

                max_ix = np.argmax(scores)
                summary_sentences.append(path_evidence[max_ix])
                summary_evidence.append(path_evidence)
                summary_evidence_paths.append(path_evidence_paths)

            # post process the extractive summaries
            summary_sentences = [
                get_true_case(sent) + ("." if sent[-1] not in ["!", "?", ".", ","] else "")
                for sent in summary_sentences
            ]

            if config.eval.metrics.hrq_agg.get("summary_dedupe_output", False):
                summary_sentences = list(set(summary_sentences))

            extractive_summs.append(" ".join(summary_sentences))
            all_evidence.append(summary_evidence)
            all_evidence_paths.append(summary_evidence_paths)

        gold_summs = [ent["summaries"] for ent in eval_data]
        pred_summs = [
            " ".join(
                list(set(sentences_by_entity[ent["entity_id"]]))
                if config.eval.metrics.hrq_agg.get("summary_dedupe_output", False)
                else sentences_by_entity[ent["entity_id"]]
            )
            for ent in eval_data
        ]

        scores = {k: v for k, v in get_jackknife_rouge(pred_summs, gold_summs).items()}

        extractive_scores = {k: v for k, v in get_jackknife_rouge(extractive_summs, gold_summs).items()}

        agent.config.eval.data["sample_outputs"] = sample_outputs

        return {"abstractive": scores, "extractive": extractive_scores}, {
            "summaries": pred_summs,
            "abstractive_sentences": [sentences_by_entity[ent["entity_id"]] for ent in eval_data],
            "paths": [summary_paths[ent["entity_id"]] for ent in eval_data],
            "weights": [summary_path_weights[ent["entity_id"]] for ent in eval_data],
            "extractive_summaries": extractive_summs,
            "evidence": all_evidence,
            "evidence_paths": all_evidence_paths,
            "paths_by_review_by_entity": paths_by_review_by_entity,
            "inputs": eval_sentences,
            "all_codes": all_codes,
        }

    @abstractmethod
    def eval_masked_generation(config, agent, test=False, dev_samples=None, test_samples=None, skip_scores=False):
        config_gen_masked = copy.deepcopy(config.data)
        config_gen_masked["dataset"] = "json"
        config_gen_masked["json_dataset"] = {
            "path": config.eval.metrics.hrq_agg.dataset_all,
            "filename": "reviews.{split}",
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "target"},
                {"type": "copy", "from": "sentence", "to": "source"},
                {"type": "copy", "from": "head_mask", "to": "head_mask"},
                {"type": "copy", "from": "residual_mask", "to": "residual_mask"},
            ],
        }

        data_loader = JsonDataLoader(
            data_path=agent.data_path,
            config=Config(config_gen_masked),
            dev_samples=dev_samples,
            test_samples=test_samples,
        )

        bneck_types = [x.type for x in agent.config.bottleneck.modules]
        if "hrqvae" not in bneck_types:
            logger.warning("Tried to run oracle masked eval on a model without a quantizer!")
            return {}
        quantizer_index = bneck_types.index("hrqvae")
        num_heads = agent.config.bottleneck.modules[quantizer_index].quantizer.num_heads

        scores = {}
        outputs = {}

        split = "test" if test else "dev"

        if not skip_scores:
            with jsonlines.open(
                os.path.join(
                    agent.data_path,
                    config.eval.metrics.hrq_agg.dataset_all,
                    f"reviews.{split}.jsonl",
                )
            ) as f:
                rows = [row for row in f][: config_gen_masked["eval"].get("truncate_dataset", None)]
            refs = [[q["sentence"]] for q in rows]

        for mask_length in range(0, num_heads + 1):
            logger.info("Generating with mask length {:}".format(mask_length))
            mask = [1] * (num_heads - mask_length) + [0] * mask_length
            samples = (data_loader._test if test else data_loader._valid).samples
            samples = [{**x, "head_mask": mask, "residual_mask": [0]} for x in samples]
            masked_loader = JsonDataLoader(
                data_path=agent.data_path, config=Config(config_gen_masked), dev_samples=samples
            )

            _, _, (output, _, _), _ = agent.inference(
                masked_loader.valid_loader, desc=f"Decoding with mask length {mask_length}"
            )

            if not skip_scores:
                # refs = [x["paras"] for x in qs_by_para_split]
                max_num_refs = max([len(x) for x in refs])
                refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

                tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded)), lowercase=True).score

                scores[mask_length] = tgt_bleu

            outputs[mask_length] = output

        return scores, outputs

    @abstractmethod
    def eval_cluster_purity(config, agent, test=False):
        sents_by_code, outputs_with_codes = HRQAggregationMetricHook.codes_from_cache(config, agent, test)

        bneck_types = [x.type for x in agent.config.bottleneck.modules]
        if "hrqvae" not in bneck_types:
            logger.warning("Tried to run oracle masked eval on a model without a quantizer!")
            return {}
        quantizer_index = bneck_types.index("hrqvae")
        num_heads = agent.config.bottleneck.modules[quantizer_index].quantizer.num_heads

        scores = {}
        for mask_len in range(num_heads):
            sents_by_cluster = defaultdict(set)
            for codes, sents in sents_by_code.items():
                sents_by_cluster[eval(codes)[: mask_len + 1]].update(sents)

            inters = []
            intras = []

            for codes, sents in tqdm(list(sents_by_cluster.items())[:2000], disable=agent.silent):
                sents = list(sents)
                if len(sents) == 1:
                    continue
                other_sents = [
                    sent for other_codes, sents in sents_by_cluster.items() if codes != other_codes for sent in sents
                ]
                np.random.shuffle(other_sents)
                other_sents = other_sents[:1000]
                inter_refs = [other_sents] * len(sents[:1000])
                inter_score = sacrebleu.corpus_bleu(sents[:1000], list(zip(*inter_refs)), lowercase=True).score
                #     inter_rouge = get_jackknife_rouge(sents[:1000], inter_refs)['rouge2']
                #     inter_rouges.append(inter_rouge)

                inters.append(inter_score)

                refs = [[s for s in sents[:100] if s != sent] for sent in sents[:100]]
                intra_score = sacrebleu.corpus_bleu(sents[:100], list(zip(*refs)), lowercase=True).score
                #     intra_rouge = get_jackknife_rouge(sents[:1000], refs)['rouge2']
                intras.append(intra_score)
            scores[num_heads - mask_len] = (np.mean(intras), np.mean(inters))

        return scores

    @abstractmethod
    def eval_oracle_summaries(config, agent, test=False, eval_data=None):
        split = "test" if test else "dev"

        if eval_data is None:
            dataset_eval = config.eval.metrics.hrq_agg.get("dataset_eval", "opagg/space-eval")
            with jsonlines.open(os.path.join(agent.data_path, dataset_eval, f"{split}.jsonl")) as reader:
                eval_data = [x for x in reader]

        reference_sentences = [
            {"sentence": sentence, "summ_id": summ_id, "entity_id": row["entity_id"]}
            for row in eval_data
            for summ_id, summ in enumerate(row["summaries"])
            for sentence in sent_tokenize(summ)
        ]

        review_sentences = [
            {"sentence": sentence, "summ_id": summ_id, "entity_id": row["entity_id"]}
            for row in eval_data
            for summ_id, summ in enumerate(row["reviews"])
            for sentence in summ["sentences"]
        ]

        # First, get the oracle codes for the reference summaries
        config_codes = copy.deepcopy(config.data)
        config_codes["dataset"] = "json"
        config_codes["json_dataset"] = {
            "path": None,
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "target"},
                {"type": "copy", "from": "sentence", "to": "source"},
            ],
        }

        data_loader = JsonDataLoader(
            config=Config(config_codes), data_path=agent.data_path, dev_samples=reference_sentences
        )

        _, _, _, memory = agent.inference(
            data_loader.valid_loader, memory_keys_to_return=["vq_codes"], desc="Getting reference encodings"
        )

        reference_codes = memory["vq_codes"].tolist()

        data_loader = JsonDataLoader(
            config=Config(config_codes), data_path=agent.data_path, dev_samples=review_sentences
        )

        _, _, _, memory = agent.inference(
            data_loader.valid_loader, memory_keys_to_return=["vq_codes"], desc="Getting review encodings"
        )

        review_codes = memory["vq_codes"].tolist()

        # Now, decode partial codes, to check how much detail is actually required
        masked_sents = []

        bneck_types = [x.type for x in agent.config.bottleneck.modules]
        quantizer_index = bneck_types.index("hrqvae")
        num_heads = agent.config.bottleneck.modules[quantizer_index].quantizer.num_heads

        config_masked = copy.deepcopy(config.data)
        config_masked["dataset"] = "json"
        config_masked["json_dataset"] = {
            "path": None,
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "target"},
                {"type": "copy", "from": "sentence", "to": "source"},
                {"type": "copy", "from": "forced_codes", "to": "forced_codes"},
                {"type": "copy", "from": "head_mask", "to": "head_mask"},
                {"type": "copy", "from": "residual_mask", "to": "residual_mask"},
            ],
        }

        for mask_length in range(num_heads - 1, 0, -1):
            mask = [1] * (num_heads - mask_length) + [0] * mask_length
            samples = [
                {**x, "sentence": "", "forced_codes": reference_codes[i], "head_mask": mask, "residual_mask": [0]}
                for i, x in enumerate(reference_sentences)
            ]

            masked_loader = JsonDataLoader(
                config=Config(config_masked), data_path=agent.data_path, dev_samples=samples
            )
            _, _, (output, _, _), _ = agent.inference(masked_loader.valid_loader, desc="Generating")
            masked_sents.append(output)
        masked_sents = list(zip(*masked_sents))

        # Determine best depth for oracle generation
        all_sims = []
        best_matches_unconstrained = []
        for inputs, output, codes in zip(reference_sentences, masked_sents, reference_codes):
            sims = [
                sacrebleu.sentence_bleu(output[i], [inputs["sentence"]], lowercase=True).score
                for i in range(num_heads - 1)
            ]

            all_sims.append(sims)

            best_matches_unconstrained.append(
                (
                    inputs["sentence"],
                    # output[np.argmax([1 if x > np.max(sims) * 0.9 else 0 for x in sims])],
                    output[np.argmax(sims)],
                    max(sims),
                    codes[: np.argmax(sims) + 1],
                )
            )

        # Finally, construct the summaries and score them
        oracle_summaries = defaultdict(lambda: defaultdict(list))
        for input_row, best_match in zip(reference_sentences, best_matches_unconstrained):
            oracle_summaries[input_row["entity_id"]][input_row["summ_id"]].append(best_match)

        scores = []
        best_summaries = []
        for row in eval_data:
            scores.append([])
            for summ_id, _ in enumerate(row["summaries"]):
                rouge = get_jackknife_rouge(
                    [" ".join([x[1] for x in oracle_summaries[row["entity_id"]][summ_id]])], [row["summaries"]]
                )["rouge2"]
                scores[-1].append(rouge)

            best_summaries.append(oracle_summaries[row["entity_id"]][np.argmax(scores[-1])])

        best_score = np.mean([np.max(s) for s in scores])

        # Now get the best summaries with limited depths
        truncated_scores = {}
        summaries_by_depth = []
        for max_depth in range(1, num_heads + 1):
            best_matches = []
            for inputs, output, codes in zip(reference_sentences, masked_sents, reference_codes):
                sims = [
                    sacrebleu.sentence_bleu(output[i], [inputs["sentence"]], lowercase=True).score
                    for i in range(num_heads - 1)
                ]

                best_matches.append(
                    (
                        inputs["sentence"],
                        # output[np.argmax([1 if x > np.max(sims) * 0.9 else 0 for x in sims[:max_depth]])],
                        output[np.argmax(sims[:max_depth])],
                        max(sims[:max_depth]),
                        codes[: np.argmax(sims[:max_depth]) + 1],
                    )
                )

            # construct the summaries and score them
            oracle_summaries = defaultdict(lambda: defaultdict(list))
            for input_row, best_match in zip(reference_sentences, best_matches):
                oracle_summaries[input_row["entity_id"]][input_row["summ_id"]].append(best_match)

            scores = []
            summaries_this_depth = []
            for row in eval_data:
                scores.append([])
                for summ_id, _ in enumerate(row["summaries"]):
                    rouge = get_jackknife_rouge(
                        [" ".join([x[1] for x in oracle_summaries[row["entity_id"]][summ_id]])], [row["summaries"]]
                    )["rouge2"]
                    scores[-1].append(rouge)
                summaries_this_depth.append(oracle_summaries[row["entity_id"]][np.argmax(scores[-1])])
            summaries_by_depth.append(summaries_this_depth)

            truncated_scores[max_depth] = np.mean([np.max(s) for s in scores])

        return (
            best_score,
            truncated_scores,
            best_summaries,
            best_matches_unconstrained,
            summaries_by_depth,
            reference_sentences,
            masked_sents,
            reference_codes,
            review_sentences,
            review_codes,
        )
