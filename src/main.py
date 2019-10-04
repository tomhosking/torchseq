from absl import app
from args import FLAGS as FLAGS

import torch

import os, json

from datasets import loaders
from datasets import cqa_triple

from agents.aq_agent import AQAgent

from utils.config import Config

from utils.bpe_factory import BPE

def main(_):

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    config = Config({
        'name': 'default',
        'log_interval': 10,
        'cuda': True,
        'seed': 0,
        'lr': 1e-4,
        'batch_size': 32,
        'eval_batch_size': 16,
        'data_path': './data',
        'gpu_device': 0,
        'embedding_dim': 300,
        'bio_embedding_dim': 12,
        'vocab_size': 10000,
        'clip_gradient': 5,
        'num_epochs': 30,
        'opt': 'adam',
        'dropout': 0.3,
        'encdec': {
            'num_encoder_layers': 4,
            'num_decoder_layers': 4,
            'num_heads': 12,
            'dim_feedforward': 512
        }
    })

    BPE.pad_id = config.vocab_size


    agent = AQAgent(config)

    try:
        if FLAGS.load_chkpt is not None:
            agent.load_checkpoint(FLAGS.load_chkpt)

        if FLAGS.train:
            agent.train()
        if FLAGS.validate:
            agent.validate(save=False)

    except KeyboardInterrupt:
        agent.logger.info("You have entered CTRL+C.. Wait to finalize")
    agent.finalize()



if __name__ == '__main__':
  app.run(main)
