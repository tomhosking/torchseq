from torchseq.eval.recipes import EvalRecipe
from torchseq.utils.model_loader import model_from_path
from torchseq.metric_hooks.opsumm_cluster_aug import OpSummClusterAugMetricHook
from torchseq.utils.timer import Timer


class Recipe(EvalRecipe):
    name: str = "opagg.extractive_summaries"

    def run(self):
        result = {}

        instance = model_from_path(self.model_path, use_cuda=(not self.cpu))

        with Timer(template="\tTime: {:.3f} seconds", show_readout=False) as t:
            scores, res = OpSummClusterAugMetricHook.eval_extract_summaries_and_score(
                self.config, instance, test=self.test
            )
        print("\tExtractive R2 = {:0.2f}".format(scores["extractive"]["rouge2"]))
        print("\tSC_ins = {:0.2f}".format(scores["extractive"]["sc_ins"]))
        print("\tSC_refs = {:0.2f}".format(scores["extractive"]["sc_refs"]))
        clustering_time = t.time

        with Timer(template="\tTime: {:.3f} seconds", show_readout=False) as t:
            score = OpSummClusterAugMetricHook.eval_compare_selected_clusters_to_oracle(
                self.config, instance, res["evidence"], test=self.test
            )
        print(
            "\tARI: {:0.3f}, (oracle {:0.1f} vs pred {:0.1f})".format(
                score["ari"], score["oracle_mean_size"], score["pred_mean_size"]
            )
        )
        ari_time = t.time

        with Timer(template="\tTime: {:.3f} seconds", show_readout=False) as t:
            prev_scores = OpSummClusterAugMetricHook.eval_cluster_prevalence(
                self.config, instance, res["evidence"], test=self.test
            )
        print("\tPrev: ", prev_scores)
        prevalence_time = t.time

        return result
