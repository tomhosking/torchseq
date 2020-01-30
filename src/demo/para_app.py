
import sys
sys.path.insert(0, './src/')

from absl import app as absl_app
from args import FLAGS as FLAGS
from flask import Flask, current_app, request, redirect, Response

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
    ctxt = request.args['ctxt']
    ans = request.args['ans']

    query = {
        's1': s1,
        'c': ctxt,
        'a': ans,
        'a_pos': ctxt.find(ans),
        'q': s1
    }
    res, scores = app.agent.infer(query, reduce_outputs=False)

    scores = scores.tolist()

    output = [list(zip(res[ix], scores[ix])) for ix in range(len(res))]

    return Response(json.dumps(output, indent=2), mimetype='application/json') 

@app.route("/api/ping")
def ping():
    return "ack"

def init():
    # Get the config
    MODEL_PATH = './runs/paraphrase/20200128_095215_parabank-qs_supp1.0_8heads'

    # with open('./runs/paraphrase/20200110_112727_kaggle_3x3/config.json') as f:
    with open(MODEL_PATH + '/config.json') as f:
        cfg_dict = json.load(f)
        cfg_dict['eval']['sampler'] = "nucleus"
        cfg_dict['training']['dataset'] = "squad"
        cfg_dict['nucleus_sampling'] = {
            "beam_width": 32,
            "cutoff": 0.9,
            "length_alpha": 0
        }
        cfg_dict['beam_search'] = {
            "beam_width": 32,
            "beam_expansion": 8,
            "length_alpha": 1.0
        }
        cfg_dict['reranker'] = {
            'strategy': 'qa'
        }
        config = Config(cfg_dict)
    
    # checkpoint_path = './runs/paraphrase/20200110_112727_kaggle_3x3/model/checkpoint.pth.tar'
    checkpoint_path = MODEL_PATH + '/model/checkpoint.pth.tar'
    

    app.agent = ParaphraseAgent(config=config, run_id=None)

    app.agent.load_checkpoint(checkpoint_path)
    app.agent.model.eval()

def main(_):
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5005, processes=1)

if __name__ == '__main__':
    absl_app.run(main)
    