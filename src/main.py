from absl import app
from args import FLAGS as FLAGS

import torch

import os, json

from datasets import loaders
from datasets import cqa_triple

from agents.aq_agent import AQAgent
from agents.prepro import PreprocessorAgent

from utils.config import Config
from utils.seed import set_seed
from utils.bpe_factory import BPE

from datetime import datetime

def main(_):

    use_cuda = torch.cuda.is_available()
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    with open(FLAGS.config) as f:
        config = Config(json.load(f))

    set_seed(FLAGS.seed)

    # This is not a good way of passing this value in
    BPE.pad_id = config.prepro.vocab_size
    BPE.embedding_dim = config.embedding_dim

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + config.name

    
    if FLAGS.preprocess:
        preprocessor = PreprocessorAgent(config)
        preprocessor.run()
        return

    agent = AQAgent(config, run_id)
    if FLAGS.load_chkpt is not None:
        agent.load_checkpoint(FLAGS.load_chkpt)

    if FLAGS.train:
        agent.train()
    if FLAGS.validate:
        agent.validate(save=False, force_save_output=True)

    




if __name__ == '__main__':
  app.run(main)
