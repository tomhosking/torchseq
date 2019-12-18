# TorchAQ

Neural Text-to-Text question generation, in pytorch


## Setup

From your venv, install requirements with `pip install -r requirements.txt`, then run `./scripts/setup.sh` to fetch external data.

## Todo

  - [x] BERT for encoder
  - [x] Better tokenisation/embeddings
  - [ ] Better fusing - can we use a bidaf style approach? or just concat a la BERT?
  - [ ] MSMarco - check how badly formed it is first?
  - [x] NewsQA
  - [ ] Other datasets? ELI5? Natural questions?
  - [ ] Expand dataset with para stuff
  - [ ] Cache internal key,value pairs from old timesteps (See Wolf et al 2019)


### Model

  - [x] S2S working
  - [x] How to add the answer into the encoding?

### Externally
  
  - [x]  start with a q autoencoder as test
  - [ ]  sample (nucleus?)
  - [x]  handle teacher forcing
  - [x]  calc loss carefully!!!
  - [-]  additional losses? regularisation? penalty for including ans in q?
  - [x]  log it all
  - [x]  grad clipping
  - [x]  checkpointing
  - [x]  config! and run management
  - [ ] load config from chkpt
  - [ ] easier eval
  - [x] log output somewhere

### Bugs

  - [x] numbers not handled by BPEmb
  - [x] double width chars
