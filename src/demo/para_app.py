import json
import sys

from absl import app as absl_app
from flask import Flask, Response, current_app, redirect, request

from agents.para_agent import ParaphraseAgent
from args import FLAGS as FLAGS
from utils.config import Config

sys.path.insert(0, "./src/")






app = Flask(__name__)


@app.route("/")
def index():
    return redirect("/static/demo.htm")


@app.route("/api/generate")
def generate():

    s1 = request.args["s1"]

    s1 = "return palm house ; return #1 that has subtropic plants from around the world on display ; return where is #1 from"
    # ctxt = request.args['ctxt']
    # ans = request.args['ans']

    query = {"s1": s1, "c": s1, "a": ";", "a_pos": s1.find(";"), "q": s1, "is_para": True}
    res, scores = app.agent.infer(query, reduce_outputs=False)

    scores = scores.tolist()

    output = [list(zip(res[ix], scores[ix])) for ix in range(len(res))]

    return Response(json.dumps(output, indent=2), mimetype="application/json")


@app.route("/api/ping")
def ping():
    return "ack"


def init():
    # Get the config
    # MODEL_PATH = './runs/paraphrase/20200203_074446_parabank_to_kaggle_finetuned_8heads'
    # MODEL_PATH = './runs/paraphrase/20200209_190027_kaggle_8heads'
    # MODEL_PATH = './runs/paraphrase/20200130_161250_parabank-qs_supp1.0_8heads'
    # MODEL_PATH = './runs/paraphrase/20200211_163735_parabank_to_kaggle_to_squad_8heads'
    MODEL_PATH = "./runs/qdmr/20200304_154646_qdmr_from_q_withresidual_abs_lang"
    # MODEL_PATH = './runs/qdmr/20200221_153512_qdmr_to_q_withresidual_relative'

    # with open('./runs/paraphrase/20200110_112727_kaggle_3x3/config.json') as f:
    with open(MODEL_PATH + "/config.json") as f:
        cfg_dict = json.load(f)
        cfg_dict["env"]["data_path"] = "./data/"
        cfg_dict["eval"]["sampler"] = "beam"
        cfg_dict["eval"]["topk"] = 32
        # cfg_dict['training']['dataset'] = "squad"
        cfg_dict["nucleus_sampling"] = {"beam_width": 24, "cutoff": 0.9, "length_alpha": 0}
        cfg_dict["beam_search"] = {"beam_width": 8, "beam_expansion": 2, "length_alpha": 0.0}
        cfg_dict["reranker"] = {
            # 'strategy': 'qa'
            "strategy": None
        }
        config = Config(cfg_dict)

    # checkpoint_path = './runs/paraphrase/20200110_112727_kaggle_3x3/model/checkpoint.pth.tar'
    checkpoint_path = MODEL_PATH + "/model/checkpoint.pth.tar"

    app.agent = ParaphraseAgent(config=config, run_id=None)

    app.agent.load_checkpoint(checkpoint_path)
    app.agent.model.eval()


def main(_):
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5005, processes=1)


if __name__ == "__main__":
    absl_app.run(main)
