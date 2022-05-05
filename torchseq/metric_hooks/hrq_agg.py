import jsonlines
import torch
import numpy as np
import copy
import os
import json
from abc import abstractmethod
from tqdm import tqdm

from collections import defaultdict, Counter
from torchseq.metric_hooks.base import MetricHook
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config
from torchseq.utils.functions import batchify, cos_sim

import sacrebleu
from transformers import AutoTokenizer, AutoModelForSequenceClassification


from openTSNE import TSNE

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import logging

logger = logging.getLogger("HRQAggregationMetric")


class HRQAggregationMetricHook(MetricHook):

    type = "slow"  # should be either 'live' or 'slow' - live metrics are calculated every epoch, slow metrics only for evaluation

    def __init__(self, config, src_field=None, tgt_field=None):
        super().__init__(config, src_field, tgt_field)

    def on_begin_epoch(self, use_test=False):
        self.scores = {}

    def on_batch(self, batch, logits, output, memory, use_test=False):
        pass

    def on_end_epoch(self, agent, use_test=False):

        if self.config.eval.metrics.hrq_agg.get("run_nli", False):
            logger.info("Running NLI eval")
            self.scores["hrq_agg"], _, _ = HRQAggregationMetricHook.eval_nli(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_tsne", False):
            logger.info("Running tsne eval")
            self.scores["hrq_agg"], _ = HRQAggregationMetricHook.eval_tsne(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_specialisation", False):
            logger.info("Running specialisation eval")
            self.scores["hrq_agg"] = HRQAggregationMetricHook.eval_specialisation(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_generate_summaries", False):
            logger.info("Running generation using HRQ paths")
            self.scores["hrq_agg"], _ = HRQAggregationMetricHook.eval_generate_summaries(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        if self.config.eval.metrics.hrq_agg.get("run_masked_generation", False):
            logger.info("Running generation with masking")
            self.scores["hrq_agg"], _ = HRQAggregationMetricHook.eval_masked_generation(
                self.config,
                agent,
                test=use_test,
            )
            logger.info("...done")

        return self.scores

    @abstractmethod
    def codes_from_cache(config, agent, test=False):

        split = "test" if test else "dev"
        if os.path.exists(agent.run_output_path + f"/sents_by_code_{split}.json"):
            with open(agent.run_output_path + f"/sents_by_code_{split}.json") as f:
                sents_by_code = json.load(f)
            with open(agent.run_output_path + f"/outputs_with_codes_{split}.json") as f:
                outputs_with_codes = json.load(f)
        else:
            cfg_dict = copy.deepcopy(config.data)

            cfg_dict["json_dataset"] = {
                "path": "opagg/space-filtered-all",
                "filename": "space_reviews.{split}",
                "field_map": [
                    {"type": "copy", "from": "sentence", "to": "s2"},
                    {"type": "copy", "from": "sentence", "to": "s1"},
                ],
            }

            cfg_dict["eval"]["metrics"]["hrq_agg"] = {
                "dataset_clusters": "opagg/space-filtered-clusters",
                "dataset_all": "opagg/space-filtered-all",
                "run_generate_summaries": False,
                "run_retrieval": False,
                "run_masked_generation": True,
            }
            cfg_dict["eval"]["eval_batch_size"] = 32
            # cfg_dict['eval']['truncate_dataset'] = 10000

            config_forced = Config(cfg_dict)

            # checkpoint_path = path_to_model + "/model/checkpoint.pt"
            # instance = ParaphraseAgent(config=config, run_id=None, output_path=None, data_path='../../data/', silent=False, verbose=False, training_mode=False)
            # instance.load_checkpoint(checkpoint_path)
            # instance.model.eval()

            data_loader = JsonDataLoader(config_forced, data_path=agent.data_path)

            _, _, (pred_output, _, _), memory = agent.inference(
                data_loader.valid_loader, memory_keys_to_return=["vq_codes"]
            )

            with jsonlines.open(agent.data_path + "/opagg/space-filtered-all/space_reviews.dev.jsonl") as f:
                inputs = [x["sentence"] for x in f]
            # with jsonlines.open(agent.data_path + "/opagg/space-filtered-all/space_reviews.dev.jsonl") as f:
            #     scores = [x["rating"] for x in f]

            sents_by_code = defaultdict(set)
            for sentence, codes in zip(inputs, memory["vq_codes"]):
                sents_by_code[str(tuple(codes.tolist()))].add(sentence)

            for k, v in sents_by_code.items():
                sents_by_code[k] = list(v)

            outputs_with_codes = {"outputs": pred_output, "codes": memory["vq_codes"].tolist()}

            with open(agent.run_output_path + f"/sents_by_code_{split}.json", "w") as f:
                json.dump(sents_by_code, f)
            with open(agent.run_output_path + f"/outputs_with_codes_{split}.json", "w") as f:
                json.dump(outputs_with_codes, f)

        return sents_by_code, outputs_with_codes

    @abstractmethod
    def eval_nli(config, agent, test=False, show_plot=False):

        split = "test" if test else "dev"

        cfg_dict = copy.deepcopy(config.data)

        # cfg_dict = config_nli.data

        cfg_dict["json_dataset"] = {
            "path": "opagg/space-filtered-all",
            "filename": "space_reviews.{split}",
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "s2"},
                {"type": "copy", "from": "sentence", "to": "s1"},
            ],
        }

        cfg_dict["eval"]["metrics"]["hrq_agg"] = {
            "dataset_clusters": "opagg/space-filtered-clusters",
            "dataset_all": "opagg/space-filtered-all",
            "run_generate_summaries": False,
            "run_retrieval": False,
            "run_masked_generation": True,
        }
        cfg_dict["eval"]["eval_batch_size"] = 32
        cfg_dict["eval"]["truncate_dataset"] = 1000

        config_nli = Config(cfg_dict)

        data_loader = JsonDataLoader(config_nli, data_path=agent.data_path)

        # checkpoint_path = path_to_model + "/model/checkpoint.pt"
        # instance = ParaphraseAgent(config=config, run_id=None, output_path=None, data_path='../../data/', silent=True, verbose=False, training_mode=False)
        # instance.load_checkpoint(checkpoint_path)
        # instance.model.eval()

        config_forced = copy.deepcopy(config_nli.data)
        config_forced["dataset"] = "json"
        config_forced["json_dataset"] = {
            "path": config.eval.metrics.hrq_agg.dataset_all,
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "s2"},
                {"type": "copy", "from": "sentence", "to": "s1"},
                {"type": "copy", "from": "head_mask", "to": "head_mask"},
            ],
        }

        num_heads = config_nli.bottleneck.modules[0].quantizer.num_heads

        samples = data_loader._valid.samples

        output_masked = {}
        probs_masked = {}
        preds_by_depth = {}

        inputs = [x["sentence"] for x in samples]

        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base-mnli")

        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base-mnli").cuda()

        # from sentence_transformers import SentenceTransformer

        # model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

        # inputs_embedded = model.encode(inputs)

        print("HRQ fwd pass")
        mean_lens = {}
        for mask_len in tqdm(range(0, num_heads)):
            mask = [1] * (num_heads - mask_len) + [0] * mask_len
            masked_samples = [{**x, "head_mask": mask} for x in samples]

            forced_loader = JsonDataLoader(config=Config(config_forced), dev_samples=masked_samples)

            _, _, (output, _, _), _ = agent.inference(forced_loader.valid_loader)

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
                [(x == 2) * 1.0 for x in preds_by_depth[tgt_mask_len]["inputs"]], axis=0
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
                    [(x == 2) * 1.0 for x in preds_by_depth[tgt_mask_len][src_mask_len]], axis=0
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
        sents_by_code, outputs_with_codes = HRQAggregationMetricHook.codes_from_cache(config, agent, test)

        split = "test" if test else "dev"

        LIMIT = 10000
        PLOT_LIMIT = 1000
        # LIMIT = 1000000

        # with open(path_to_model + "/config.json") as f:
        #     cfg_dict = json.load(f)
        # from torchseq.utils.config import Config

        # config = Config(cfg_dict)

        # with open(path_to_model + 'outputs_with_codes_dev.json') as f:
        #     outputs_with_codes = json.load(f)

        codes = [tuple(x) for x in outputs_with_codes["codes"]][:LIMIT]
        outputs = outputs_with_codes["outputs"][:LIMIT]

        # chkpt = torch.load(path_to_model + "model/checkpoint.pt", map_location='cpu')

        num_heads = config.bottleneck.modules[0].quantizer.num_heads

        # embeddings = [torch.nn.Embedding.from_pretrained(chkpt['model_state_dict'][f'bottleneck.module_list.0.quantizer._embedding.{hix}.weight']).cpu() for hix in range(num_heads)]
        embeddings = agent.model.bottleneck.module_list[0].quantizer._embedding

        embedded_codes = (
            torch.cat(
                [
                    torch.cat(
                        [embeddings[hix](torch.LongTensor([x[hix]]).to(agent.device)) for hix in range(num_heads)],
                        dim=0,
                    ).unsqueeze(0)
                    for x in codes
                ],
                dim=0,
            )
            .detach()
            .cpu()
        )
        partial_codes = torch.cat(
            [torch.sum(embedded_codes[:, :(hix), :], dim=1, keepdim=True) for hix in range(num_heads + 1)], dim=1
        )
        full_embeddings = torch.sum(embedded_codes, dim=1)

        with jsonlines.open(agent.data_path + f"/opagg/space-filtered-all/space_reviews.{split}.jsonl") as f:
            entity_ids = [x["entity_id"] for x in f][:LIMIT]
        with jsonlines.open(agent.data_path + f"/opagg/space-filtered-all/space_reviews.{split}.jsonl") as f:
            review_ids = [x["review_id"] for x in f][:LIMIT]

        # construct probabilistic trees

        trees_by_entity = defaultdict(lambda: defaultdict(int))
        trees_by_entity_probs = defaultdict(lambda: defaultdict(float))

        for entity_id, review_id, path, pred_sent in zip(entity_ids, review_ids, codes, outputs):
            for h in range(num_heads):
                trees_by_entity[entity_id][path[: h + 1]] += 1

        # normalise
        for entity_id in trees_by_entity.keys():
            for h in range(num_heads):
                total = sum([v for k, v in trees_by_entity[entity_id].items() if len(k) == (h + 1)])
                for k, v in trees_by_entity[entity_id].items():
                    if len(k) == (h + 1):
                        trees_by_entity_probs[entity_id][k] = 1.0 * v / total

        tsne = TSNE(n_components=2)

        X_full_embedded = tsne.fit(full_embeddings)

        X_byhead_embedded = X_full_embedded.transform(partial_codes.reshape(-1, 768)).reshape(-1, num_heads + 1, 2)

        colors = list(mcolors.XKCD_COLORS.values())
        np.random.shuffle(colors)

        color_labels = [colors[x[0]] for x in codes]

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

        linecols = ["tab:blue", "red", "blue", "orange", "grey", "grey", "grey", "grey"]

        plt.figure(figsize=(18, 12))
        ax = plt.gca()

        unique_entities = list(set(entity_ids))

        markers = [marker_types[codes[1] % len(marker_types)] for codes in codes]
        patterns = [pattern_types[codes[2] % len(pattern_types)] for codes in codes]
        for i, (x, y, c, m, p) in enumerate(
            zip(X_full_embedded.T[0], X_full_embedded.T[1], color_labels, markers, patterns)
        ):
            if (
                unique_entities.index(entity_ids[i]) > 2 or unique_entities.index(entity_ids[i]) < 1
            ) and i > PLOT_LIMIT:
                continue
            ax.scatter(x, y, color=c, s=10, marker=m)  # hatch=4*p, facecolor='white'

        entitycols = [linecols[unique_entities.index(x) % len(linecols)] for x in entity_ids]

        plt.title(agent.run_id)

        for hix in range(0, 2):
            for i in range(LIMIT):
                if unique_entities.index(entity_ids[i]) > 2 or unique_entities.index(entity_ids[i]) < 1:
                    continue
                from_coords = X_byhead_embedded[i : (i + 1), hix, :]
                to_coords = X_byhead_embedded[i : (i + 1), hix + 1, :]
                ab_pairs = np.c_[from_coords, to_coords]
                ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)
                ax.plot(*ab_args, c=entitycols[i], linewidth=3, alpha=0.05)
        plt.savefig(agent.run_output_path + "/tsne_entity_overlay.pdf", bbox_inches="tight")

        if show_plot:
            plt.show()

        return {}, []

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

        with jsonlines.open(agent.data_path + f"/opagg/space-filtered-all/space_reviews.{split}.jsonl") as f:
            entity_ids = [x["entity_id"] for x in f]
        with jsonlines.open(agent.data_path + f"/opagg/space-filtered-all/space_reviews.{split}.jsonl") as f:
            review_ids = [x["review_id"] for x in f]
        with jsonlines.open(agent.data_path + f"/opagg/space-filtered-all/space_reviews.{split}.jsonl") as f:
            inputs = [x["sentence"] for x in f]
        with jsonlines.open(agent.data_path + f"/opagg/space-filtered-all/space_reviews.{split}.jsonl") as f:
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

        # Measures of tree concentration

        # entropy (tree | entity_id)

        # construct probabilistic trees
        trees_by_entity = defaultdict(lambda: defaultdict(int))
        trees_by_review = defaultdict(lambda: defaultdict(int))
        trees_by_entity_probs = defaultdict(lambda: defaultdict(float))
        trees_by_review_probs = defaultdict(lambda: defaultdict(float))

        for entity_id, review_id, path in zip(entity_ids, review_ids, codes):
            for h in range(num_heads):
                trees_by_entity[entity_id][path[: h + 1]] += 1
                trees_by_review[review_id][path[: h + 1]] += 1

        # normalise
        entropies_entity_by_head = {}
        entropies_review_by_head = {}

        for h in range(num_heads):
            entropies_entity_by_head[h] = []
            entropies_review_by_head[h] = []

            for entity_id in trees_by_entity.keys():
                total = sum([v for k, v in trees_by_entity[entity_id].items() if len(k) == (h + 1)])
                for k, v in trees_by_entity[entity_id].items():
                    if len(k) == (h + 1):
                        trees_by_entity_probs[entity_id][k] = 1.0 * v / total
                this_probs = [prob for k, prob in trees_by_entity_probs[entity_id].items() if len(k) == (h + 1)]
                this_entropy = np.sum(-1.0 * np.log(this_probs) * this_probs)
                entropies_entity_by_head[h].append(this_entropy)

            entropies_entity_by_head[h] = np.mean(entropies_entity_by_head[h])

            for review_id in trees_by_review.keys():
                total = sum([v for k, v in trees_by_review[review_id].items() if len(k) == (h + 1)])
                for k, v in trees_by_review[review_id].items():
                    if len(k) == (h + 1):
                        trees_by_review_probs[review_id][k] = 1.0 * v / total
                this_probs = [prob for k, prob in trees_by_review_probs[review_id].items() if len(k) == (h + 1)]
                this_entropy = np.sum(-1.0 * np.log(this_probs) * this_probs)
                entropies_review_by_head[h].append(this_entropy)

            entropies_review_by_head[h] = np.mean(entropies_review_by_head[h])

        uniform_by_depth = {}
        for d in range(num_heads):
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
    def eval_generate_summaries(config, agent, test=False):
        # First, get encodings for all sentences
        config_codes = copy.deepcopy(config.data)
        config_codes["dataset"] = "json"
        config_codes["json_dataset"] = {
            "path": config.eval.metrics.hrq_agg.dataset_all,
            "filename": "space_reviews.{split}",
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "s2"},
                {"type": "copy", "from": "sentence", "to": "s1"},
            ],
        }

        MASK_LENGTH = 3

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

        data_loader = JsonDataLoader(config=Config(config_codes), data_path=agent.data_path)

        sample_outputs = agent.config.eval.get("sample_outputs", True)
        agent.config.eval.data["sample_outputs"] = False

        _, _, _, memory = agent.inference(
            data_loader.test_loader if test else data_loader.valid_loader, memory_keys_to_return=["vq_codes"]
        )

        all_codes = memory["vq_codes"]

        split = "test" if test else "dev"

        with jsonlines.open(
            os.path.join(agent.data_path, config.eval.metrics.hrq_agg.dataset_all, f"space_reviews.{split}.jsonl")
        ) as reader:
            all_rows = [x for x in reader]

        # Identify top clusters per entity
        codes_by_entity = defaultdict(Counter)

        for row, codes in zip(all_rows, all_codes):
            codes_by_entity[row["entity_id"]][tuple(codes.tolist()[:-MASK_LENGTH])] += 1

        mask = [1] * (num_heads - MASK_LENGTH) + [0] * MASK_LENGTH

        filtered_examples = []
        for entity, counter in codes_by_entity.items():
            for codes, count in counter.most_common(5):
                filtered_examples.append(
                    {"entity_id": entity, "codes": list(codes) + [0] * MASK_LENGTH, "sentence": "", "head_mask": mask}
                )

        # Generate!
        agent.config.eval.data["sample_outputs"] = True

        config_forced = copy.deepcopy(config.data)
        config_forced["dataset"] = "json"
        config_forced["json_dataset"] = {
            "path": config.eval.metrics.hrq_agg.dataset_all,
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "s2"},
                {"type": "copy", "from": "sentence", "to": "s1"},
                {"type": "copy", "from": "codes", "to": "forced_codes"},
                {"type": "copy", "from": "head_mask", "to": "head_mask"},
            ],
        }

        forced_loader = JsonDataLoader(config=Config(config_forced), dev_samples=filtered_examples)

        _, _, (output, _, _), _ = agent.inference(forced_loader.valid_loader)

        # TODO: eval against reference summaries

        agent.config.eval.data["sample_outputs"] = sample_outputs

        return {}, output

    @abstractmethod
    def eval_masked_generation(config, agent, test=False, dev_samples=None, test_samples=None, skip_scores=False):
        config_gen_masked = copy.deepcopy(config.data)
        config_gen_masked["dataset"] = "json"
        config_gen_masked["json_dataset"] = {
            "path": config.eval.metrics.hrq_agg.dataset_all,
            "filename": "space_reviews.{split}",
            "field_map": [
                {"type": "copy", "from": "sentence", "to": "s2"},
                {"type": "copy", "from": "sentence", "to": "s1"},
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
                    f"space_reviews.{split}.jsonl",
                )
            ) as f:
                rows = [row for row in f][: config_gen_masked["eval"].get("truncate_dataset", None)]
            refs = [[q["sentence"]] for q in rows]

        for mask_length in range(0, num_heads + 1):
            mask = [1] * (num_heads - mask_length) + [0] * mask_length
            samples = (data_loader._test if test else data_loader._valid).samples
            samples = [{**x, "head_mask": mask, "residual_mask": [0]} for x in samples]
            masked_loader = JsonDataLoader(
                data_path=agent.data_path, config=Config(config_gen_masked), dev_samples=samples
            )

            _, _, (output, _, _), _ = agent.inference(masked_loader.valid_loader)

            if not skip_scores:
                # refs = [x["paras"] for x in qs_by_para_split]
                max_num_refs = max([len(x) for x in refs])
                refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

                tgt_bleu = sacrebleu.corpus_bleu(output, list(zip(*refs_padded)), lowercase=True).score

                scores[mask_length] = tgt_bleu

            outputs[mask_length] = output

        return scores, outputs
