# TorchAQ

Neural Text-to-Text question generation, in pytorch


## Setup

From your venv, install requirements with `pip install -r requirements.txt`, then run `./setup.sh` to fetch external data.

## Todo


### Model

  - [ ]  get embedded version of context
  - [ ]  augment it (esp. with ans BIO)
  - [x]  positional encoding
  - []  encode
  - [ ]  (do we need to do anything with this encoding?)
  - [ ]  decode
  - [ ]  check additional output/projection layers that are required
  - [ ]  check alignment of in/out seqs
  - [ ]  (copy mech?)

### Externally
  
  - [x]  start with a q autoencoder as test
  - [ ]  sample (nucleus?)
  - [x]  handle teacher forcing
  - [ ]  calc loss carefully!!!
  - [ ]  additional losses? regularisation? penalty for including ans in q?
  - [ ]  log it all
  - [ ]  grad clipping
  - [ ]  checkpointing
  - [ ]  config! and run management

### Bugs

  - [ ] numbers not handled by BPEmb
