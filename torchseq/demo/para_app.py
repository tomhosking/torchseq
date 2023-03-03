import json
import sys

# sys.path.insert(0, "./src/")

from flask import Flask, Response, current_app, redirect, request

from torchseq.agents.seq2seq_agent import Seq2SeqAgent
from torchseq.datasets.json_loader import JsonDataLoader

from torchseq.utils.config import Config
from torchseq.utils.tokenizer import Tokenizer


sys.path.append("/home/tom/dev/truecase/")
import truecase


app = Flask(__name__)


@app.route("/")
def index():
    return redirect("/static/para_demo.htm")


@app.route("/api/generate")
def generate():
    s1 = request.args["sem_input"]

    # s1 = "(l-punct (l-nsubj (l-cop w-1-what w-2-is) (l-acl (l-det (l-BIND w-4-plague v-w-4-plague) w-3-another) (l-nsubj (l-xcomp w-5-thought (l-mark (l-aux (l-dobj w-8-spread (l-det (l-amod w-11-way w-10-same) w-9-the)) w-7-have) w-6-to)) v-w-4-plague))) w-12-?)"
    # ctxt = request.args['ctxt']
    # ans = request.args['ans']

    # query = {"source": s1, "c": s1, "a": ";", "a_pos": 0, "q": s1, "target": s1, "sem_input": s1, "tgt": s1, "syn_input": s1}
    query = {"sem_input": s1, "tgt": "", "syn_input": s1, "question": "?", "answer": s1}
    if "template" in request.args:
        template = request.args["template"]
        print("Template: ", template)
        query["template"] = template
        query["syn_input"] = template

    data_loader = JsonDataLoader(app.agent.config, test_samples=[query])
    loss, metrics, (pred_output, gold_output, gold_input), memory = app.agent.inference(data_loader.test_loader)

    # scores = scores.tolist()

    # output = [list(zip(res[ix], scores[ix])) for ix in range(len(res))]
    return truecase.get_true_case(pred_output[0]).replace("'S", "'s")

    # return Response(json.dumps(output, indent=2), mimetype="application/json")


@app.route("/api/ping")
def ping():
    return "ack"


def init():
    # Get the config
    # MODEL_PATH = (
    #     "./runs/sep_ae/20201207_224719_vae_wikianswers-triples-chunk-extendstop-realexemplars-uniform-drop10-N10-R100"

    # )

    # MODEL_PATH = "./runs/separator/20210503_125508_wa_replication"
    # MODEL_PATH = "./runs/separatorv2/20210624_132416_wa_16codes_H4_gumbel_unmatched"

    MODEL_PATH = "./models/examples/separator-wa-v1.1"

    with open(MODEL_PATH + "/config.json") as f:
        cfg_dict = json.load(f)
        # cfg_dict["task"] = "autoencoder"
        cfg_dict["eval"]["sampler"] = "beam"
        cfg_dict["eval"]["topk"] = 1
        # cfg_dict["training"]["dataset"] = "squad"
        cfg_dict["nucleus_sampling"] = {"beam_width": 12, "cutoff": 0.9, "length_alpha": 0}
        cfg_dict["beam_search"] = {"beam_width": 4, "beam_expansion": 2, "length_alpha": 1.0}
        cfg_dict["diverse_beam"] = {
            "beam_width": 16,
            "beam_expansion": 8,
            "length_alpha": 0.0,
            "num_groups": 8,
            "penalty_weight": 0.5,
        }
        # cfg_dict["reranker"] = {
        #     # 'strategy': 'qa'
        #     "strategy": "ngram"
        # }

        # cfg_dict["bottleneck"]["prior_var_weight"] = (
        #     [1.0] * var_offset + [2.5] + [2.5] * (cfg_dict["encdec"]["num_heads"] - var_offset - 1)
        # )

        # cfg_dict["json_dataset"]["field_map"] = [
        #     {"from": "syn_input", "to": "template"},
        #     {"from": "tgt", "to": "target"},
        #     {"from": "sem_input", "to": "source"},
        # ]
        # if "bottleneck" in cfg_dict:
        #     cfg_dict["bottleneck"]["prior_var_weight"] = 0.0
        #     var_offset = 0
        #     num_heads = 4
        #     cfg_dict["bottleneck"]["code_offset"] = [0] * var_offset + [0] + [0] * (num_heads - var_offset - 1)
        config = Config(cfg_dict)

    # Tokenizer(config.prepro.tokenizer)

    # checkpoint_path = './runs/paraphrase/20200110_112727_kaggle_3x3/model/checkpoint.pth.tar'
    checkpoint_path = MODEL_PATH + "/model/checkpoint.pt"

    app.agent = Seq2SeqAgent(
        config=config, run_id=None, output_path="./runs/parademo/", silent=True, verbose=False, training_mode=False
    )

    app.agent.load_checkpoint(checkpoint_path)
    app.agent.model.eval()


def main():
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5005, processes=1)


if __name__ == "__main__":
    main()
