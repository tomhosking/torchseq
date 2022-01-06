import json
import sys

# sys.path.insert(0, "./src/")

from flask import Flask, Response, current_app, redirect, request

from torchseq.agents.aq_agent import AQAgent
from torchseq.datasets.qa_loader import QADataLoader
from torchseq.utils.config import Config
from torchseq.utils.tokenizer import Tokenizer


app = Flask(__name__)


@app.route("/")
def index():
    return redirect("/static/qg_demo.htm")


@app.route("/api/generate")
def generate():

    context = request.args["context"]
    answer = request.args["answer"]
    a_pos = context.find(answer)

    query = {"c": context, "a": answer, "a_pos": a_pos, "q": ""}

    # res, scores, _ = app.agent.infer(query, reduce_outputs=False)

    data_loader = QADataLoader(app.agent.config, test_samples=[query])
    loss, metrics, (pred_output, gold_output, gold_input), memory = app.agent.inference(data_loader.test_loader)

    # scores = scores.tolist()

    output = pred_output

    return Response(json.dumps(output, indent=2), mimetype="application/json")


@app.route("/api/ping")
def ping():
    return "ack"


def init():

    # MODEL_SLUG = "20200220_161434_bert_embeds_para_pbkagsq_ft_squad"

    # MODEL_PATH = f'./runs/augmented/{MODEL_SLUG}/'
    MODEL_PATH = "./models/examples/20210222_145021_qg_bert/"

    # Get the config
    with open(MODEL_PATH + "config.json") as f:
        cfg_dict = json.load(f)

    # Override a few bits
    cfg_dict["eval"]["topk"] = 1
    # cfg_dict["reranker"] = {
    #     # 'strategy': 'qa'
    #     "strategy": None
    # }

    config = Config(cfg_dict)

    checkpoint_path = MODEL_PATH + "model/checkpoint.pt"

    # Tokenizer(config.prepro.tokenizer)

    app.agent = AQAgent(config=config, run_id=None, output_path="./runs/parademo/", training_mode=True)

    app.agent.load_checkpoint(checkpoint_path)
    app.agent.model.eval()


def main(_):
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5004, processes=1)


if __name__ == "__main__":
    main()
