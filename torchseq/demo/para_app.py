import json
import sys

# sys.path.insert(0, "./src/")

from absl import app as absl_app
from flask import Flask, Response, current_app, redirect, request

from torchseq.agents.para_agent import ParaphraseAgent

from torchseq.utils.config import Config
from torchseq.utils.tokenizer import Tokenizer


app = Flask(__name__)


@app.route("/")
def index():
    return redirect("/static/demo.htm")


@app.route("/api/generate")
def generate():

    s1 = request.args["s1"]

    # s1 = "(l-punct (l-nsubj (l-cop w-1-what w-2-is) (l-acl (l-det (l-BIND w-4-plague v-w-4-plague) w-3-another) (l-nsubj (l-xcomp w-5-thought (l-mark (l-aux (l-dobj w-8-spread (l-det (l-amod w-11-way w-10-same) w-9-the)) w-7-have) w-6-to)) v-w-4-plague))) w-12-?)"
    # ctxt = request.args['ctxt']
    # ans = request.args['ans']

    query = {"s1": s1, "c": s1, "a": ";", "a_pos": 0, "q": s1, "s2": s1}
    if "template" in request.args:
        template = request.args["template"]
        query["template"] = template
    res, scores = app.agent.infer(query, reduce_outputs=False)

    scores = scores.tolist()

    output = [list(zip(res[ix], scores[ix])) for ix in range(len(res))]

    return Response(json.dumps(output, indent=2), mimetype="application/json")


@app.route("/api/ping")
def ping():
    return "ack"


def init():
    # Get the config
    MODEL_PATH = "./runs/sep_ae/20200922_170201_vae_squad_flipped"
    # MODEL_PATH = "./runs/paraphrase/20200519_151820_ae_nqnewsqa"

    with open(MODEL_PATH + "/config.json") as f:
        cfg_dict = json.load(f)
        # cfg_dict["task"] = "autoencoder"
        cfg_dict["env"]["data_path"] = "./data/"
        cfg_dict["eval"]["sampler"] = "nucleus"
        cfg_dict["eval"]["topk"] = 32
        cfg_dict["training"]["dataset"] = "squad"
        cfg_dict["nucleus_sampling"] = {"beam_width": 12, "cutoff": 0.9, "length_alpha": 0}
        cfg_dict["beam_search"] = {"beam_width": 16, "beam_expansion": 2, "length_alpha": 0.0}
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
        # cfg_dict["encdec"]["prior_var_weight"] = 0.0
        var_offset = 4
        cfg_dict["encdec"]["prior_var_weight"] = (
            [0.0] * var_offset + [2.0] + [2.0] * (cfg_dict["encdec"]["num_heads"] - var_offset - 1)
        )
        # cfg_dict["encdec"]["code_offset"] = (
        #     [0] * var_offset + [50] + [50] * (cfg_dict["encdec"]["quantizer_heads"] - var_offset - 1)
        # )
        config = Config(cfg_dict)

    Tokenizer(config.prepro.tokenizer)

    # checkpoint_path = './runs/paraphrase/20200110_112727_kaggle_3x3/model/checkpoint.pth.tar'
    checkpoint_path = MODEL_PATH + "/model/checkpoint.pt"

    app.agent = ParaphraseAgent(config=config, run_id=None, output_path="./runs/parademo/")

    app.agent.load_checkpoint(checkpoint_path)
    app.agent.model.eval()


def main(_):
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5005, processes=1)


if __name__ == "__main__":
    absl_app.run(main)
