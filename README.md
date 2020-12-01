# TorchSeq

[![codecov](https://codecov.io/gh/tomhosking/torchseq/branch/master/graph/badge.svg?token=GK9W2LMJDU)](https://codecov.io/gh/tomhosking/torchseq)  ![TorchSeq](https://github.com/tomhosking/torchseq/workflows/TorchSeq/badge.svg) [![Documentation Status](https://readthedocs.org/projects/torchseq/badge/?version=latest)](https://torchseq.readthedocs.io/en/latest/?badge=latest)


TorchSeq is a research-first sequence modelling framework built in Pytorch. It's designed to be easy to hack, without crazy levels of inheritance and easy access to the guts of the model.

## Setup

From a fresh venv, run:

```
pip install -r requirements.txt
pip install -e .

python3 -m nltk.downloader punkt
python3 ./scripts/download_models.py
```

## Quickstart

Here's a snippet to run inference on a pre-trained paraphrasing model:

```
from torchseq.agents.para_agent import ParaphraseAgent
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config

path_to_model = '/path/to/model/'

with open(path_to_model + "/config.json") as f:
    cfg_dict = json.load(f)
config = Config(cfg_dict)

# Load the dataset
data_loader = JsonDataLoader(config)

checkpoint_path = path_to_model + "/model/checkpoint.pt"

# Create an agent
instance = ParaphraseAgent(config=config, run_id=None, output_path="./runs/demo/", silent=True, verbose=False)

# Load the checkpoint into the agent
instance.load_checkpoint(checkpoint_path)
instance.model.eval()

# Run!
loss, metrics, output, memory = instance.validate(data_loader, save_model=False)
```
