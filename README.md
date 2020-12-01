# TorchSeq

Seq2Seq framework built in Pytorch

[![codecov](https://codecov.io/gh/tomhosking/torchseq/branch/master/graph/badge.svg?token=GK9W2LMJDU)](https://codecov.io/gh/tomhosking/torchseq)  ![TorchSeq](https://github.com/tomhosking/torchseq/workflows/TorchSeq/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/torchseq/badge/?version=latest)](https://torchseq.readthedocs.io/en/latest/?badge=latest)


## Demo

Run `./torchseq/demo/app.py`

## Features

### Tasks

 - Autoencoder
 - Paraphrasing
 - Question Generation

### Models

 - Transformer S2S
 - Variational bottleneck
 - Custom copy mechanism
 - Pretrained models (BERT, BART)

### Misc

 - Token dropout
 - Ranger optimiser
 - Mish activation
 - Gradient checkpointing



## Setup

From a fresh venv, run:

```
pip install -r requirements.txt
pip install -e .

python3 -m nltk.downloader punkt
python3 ./scripts/download_models.py
```

## Todo

  - [ ] Cache internal key,value pairs from old timesteps (See Wolf et al 2019)

