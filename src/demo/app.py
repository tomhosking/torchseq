from absl import app as absl_app
from args import FLAGS as FLAGS
from flask import Flask, current_app, request, redirect

import json

from agents.aq_agent import AQAgent


from utils.config import Config

app = Flask(__name__)

@app.route("/")
def index():
    return redirect("/static/demo.htm")


@app.route("/api/generate")
def generate():

    context = request.args['context']
    answer = request.args['answer']
    a_pos = context.find(answer)

    query = {
        'c': context,
        'a': answer,
        'a_pos': a_pos
    }
    res = app.agent.infer(query)

    return res[0]

@app.route("/api/ping")
def ping():
    return "ack"

def init():
    # Get the config
    with open(FLAGS.config) as f:
        config = Config(json.load(f))
    checkpoint_path = './models/optimised/bert_fixed/0sent/model/checkpoint.pth.tar'

    app.agent = AQAgent(config=config, run_id=None)

    app.agent.load_checkpoint(checkpoint_path)

def main(_):
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5004, processes=1)

if __name__ == '__main__':
    absl_app.run(main)
    