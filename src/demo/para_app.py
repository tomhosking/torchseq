from absl import app as absl_app
from args import FLAGS as FLAGS
from flask import Flask, current_app, request, redirect

import json

from agents.para_agent import ParaphraseAgent


from utils.config import Config

app = Flask(__name__)

@app.route("/")
def index():
    return redirect("/static/demo.htm")


@app.route("/api/generate")
def generate():

    s1 = request.args['s1']

    query = {
        's1': s1
    }
    res = app.agent.infer(query)

    return json.dumps(res[0])

@app.route("/api/ping")
def ping():
    return "ack"

def init():
    # Get the config
    with open(FLAGS.config) as f:
        config = Config(json.load(f))
    checkpoint_path = './runs/paraphrase/20200110_112727_kaggle_3x3/model/checkpoint.pth.tar'

    app.agent = ParaphraseAgent(config=config, run_id=None)

    app.agent.load_checkpoint(checkpoint_path)
    app.agent.model.eval()

def main(_):
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5005, processes=1)

if __name__ == '__main__':
    absl_app.run(main)
    