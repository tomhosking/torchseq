# TorchSeq

Seq2Seq framework built in Pytorch

[![codecov](https://codecov.io/gh/tomhosking/torchseq/branch/master/graph/badge.svg?token=GK9W2LMJDU)](https://codecov.io/gh/tomhosking/torchseq)  ![TorchSeq](https://github.com/tomhosking/torchseq/workflows/TorchSeq/badge.svg)

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

From your venv, install requirements with `pip install -r requirements.txt`.

## Todo

  - [ ] Cache internal key,value pairs from old timesteps (See Wolf et al 2019)

