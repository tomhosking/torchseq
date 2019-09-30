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
        'log_interval': 10,
        'cuda': True,
        'seed': 0,
        'lr': 1e-4,
        'batch_size': 24,
        'data_path': './data',
        'gpu_device': 0,
        'embedding_dim': 200,
        'vocab_size': 10000
    })


    # # train_loader = torch.utils.data.DataLoader()
    # train_squad = loaders.load_squad_triples(path=os.path.join(FLAGS.data_path,'squad/'), dev=False, test=False)
    # vocab = loader.get_vocab("this is a test", vocab_size=3)
    # vocab = loaders.get_glove_vocab(path=FLAGS.data_path+'/', size=FLAGS.vocab_size)
    # vocab = BPE.instance().

    # print(train_squad[0])
    # trip = train_squad[0]
    # ex = cqa_triple.CQATriple(vocab, trip[0], trip[2], trip[3], trip[1])
    # ex = cqa_triple.CQATriple(vocab, "this is a text context funicular", "funicular", 23, "where is the funicular?")
    #
    # print(ex.ctxt_as_ids())
    # print(ex.q_as_ids())
    # print(ex.copy_vocab)


    agent = AQAgent(config)

    try:
        agent.train()

    except KeyboardInterrupt:
        agent.logger.info("You have entered CTRL+C.. Wait to finalize")
    agent.finalize()



if __name__ == '__main__':
  app.run(main)
