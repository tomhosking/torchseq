# TorchSeq

Seq2Seq framework built in Pytorch

## Demo

Run `./src/demo/app.py`

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

