# TorchSeq

[![codecov](https://codecov.io/gh/tomhosking/torchseq/branch/master/graph/badge.svg?token=GK9W2LMJDU)](https://codecov.io/gh/tomhosking/torchseq)  ![TorchSeq](https://github.com/tomhosking/torchseq/workflows/TorchSeq/badge.svg) [![Documentation Status](https://readthedocs.org/projects/torchseq/badge/?version=latest)](https://torchseq.readthedocs.io/en/latest/?badge=latest)


TorchSeq is a research-first sequence modelling framework built in Pytorch. It's designed to be easy to hack, without crazy levels of inheritance and easy access to the guts of the model.

## Setup

From a fresh venv, run:

```
pip install -r requirements.txt
pip install -e .
python3 ./scripts/download_models.py
```

## Quickstart

Have a look in `examples/` for some notebooks that show how to interact with pretrained models, or try training your own models using the configuration files in `configs/`:

```
torchseq --train --config ./configs/qg_bart.json
```
