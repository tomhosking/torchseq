# TorchSeq

[![codecov](https://codecov.io/gh/tomhosking/torchseq/branch/main/graph/badge.svg?token=GK9W2LMJDU)](https://codecov.io/gh/tomhosking/torchseq)  ![TorchSeq](https://github.com/tomhosking/torchseq/workflows/TorchSeq/badge.svg) [![Documentation Status](https://readthedocs.org/projects/torchseq/badge/?version=latest)](https://torchseq.readthedocs.io/en/latest/?badge=latest)


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

Then evaluate it with a snippet like this:

```
import json
from torchseq.agents.aq_agent import AQAgent
from torchseq.datasets.qa_loader import QADataLoader
from torchseq.utils.config import Config
import torch

model_path = '../runs/examples/qg_bart/'

# Load the config
with open(model_path + 'configs/aq.json') as f:
    cfg_dict = json.load(f)
cfg_dict["env"]["data_path"] = "../data/"

config = Config(cfg_dict)

# Load the model
instance = AQAgent(config=config, run_id=None, output_path="./runs/examples/qg_bart_eval", silent=False, verbose=False)
instance.load_checkpoint(model_path + 'model/checkpoint.pt')
instance.model.eval()

# Create a dataset
data_loader = QADataLoader(config)

# Run inference on the test split
test_loss, all_metrics, (pred_output, gold_output, gold_input), memory_values_to_return = instance.inference(data_loader.test_loader)

# Done!
print(all_metrics['bleu'])
```


You can also easily run your model on a custom dataset:

```
examples = [
    {'c': 'Creme Puff was the oldest cat.', 'a': 'Creme Puff'},
    {'c': 'Creme Puff lived for 38 years and 3 days', 'a': '38 years and 3 days'},
]

# The examples need the answer character position, and a placeholder for the question
examples = [
    {**ex, 'a_pos': ex['c'].index(ex['a']), 'q': ''} for ex in examples
]
    
data_loader_custom = QADataLoader(config, test_samples=examples)

test_loss, all_metrics, (pred_output, gold_output, gold_input), memory_values_to_return = instance.inference(data_loader_custom.test_loader)

print(pred_output)
```

## Pretrained models

### Question Generation

A transformer model trained on SQuAD - \[Config\] \[Download checkpoint\] \[Download dataset\] 

A transformer model using BERT encodings trained on SQuAD

BART fine tuned on SQuAD

### Paraphrasing

A VAE model trained on Paralex

## Citation

If you find TorchSeq useful, please cite one of my papers!

```
@inproceedings{hosking-riedel-2019-evaluating,
    title = "Evaluating Rewards for Question Generation Models",
    author = "Hosking, Tom  and
      Riedel, Sebastian",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1237",
    doi = "10.18653/v1/N19-1237",
    pages = "2278--2283",
}
```