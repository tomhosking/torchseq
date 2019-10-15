from absl import app
from args import FLAGS as FLAGS

import torch

import os, json

from datasets import loaders
from datasets import cqa_triple

from agents.aq_agent import AQAgent

from utils.config import Config

from utils.bpe_factory import BPE

from datetime import datetime

def main(_):

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    config = Config({
        'name': 'default_1024FF_15H_300E-fix_2x2_nums_fixout_masked',
        'log_interval': 10,
        'cuda': True,
        'seed': 0,
        'lr': 1e-4,
        'batch_size': 64,
        'eval_batch_size': 16,
        'data_path': './data',
        'gpu_device': 0,
        'embedding_dim': 300,
        'bio_embedding_dim': 8,
        'vocab_size': 25000,
        'clip_gradient': 5,
        'num_epochs': 60,
        'opt': 'adam',
        'dropout': 0.2,
        'label_smoothing': 0.1,
        'freeze_embeddings': True,
        'freeze_projection': True,
        'directional_masks': True,
        'encdec': {
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'num_heads': 15,
            'dim_feedforward': 1024
        }
    })

    # This is not a good way of passing this value in
    BPE.pad_id = config.vocab_size
    BPE.embedding_dim = config.embedding_dim

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + config.name


    agent = AQAgent(config, run_id)

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
