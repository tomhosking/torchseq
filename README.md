# TorchAQ

Neural Text-to-Text question generation, in pytorch


## Setup

From your venv, install requirements with `pip install -r requirements.txt`, then run `./setup.sh` to fetch external data.

## Todo

  - [ ] BERT for encoder
  - [ ] MSMarco
  - [ ] Other datasets?


### Model

  - [x] S2S working
  - [ ] How to add the answer into the encoding?

### Externally
  
  - [x]  start with a q autoencoder as test
  - [ ]  sample (nucleus?)
  - [x]  handle teacher forcing
  - [x]  calc loss carefully!!!
  - [ ]  additional losses? regularisation? penalty for including ans in q?
  - [x]  log it all
  - [x]  grad clipping
  - [x]  checkpointing
  - [x]  config! and run management
  - [ ] load config from chkpt
  - [x] log output somewhere

### Bugs

  - [ ] numbers not handled by BPEmb
  - [ x double width chars
