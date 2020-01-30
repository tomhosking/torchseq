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
  - [x] Expand dataset with para stuff
  - [ ] Cache internal key,value pairs from old timesteps (See Wolf et al 2019)

