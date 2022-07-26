# TorchSeq

[![codecov](https://codecov.io/gh/tomhosking/torchseq/branch/main/graph/badge.svg?token=GK9W2LMJDU)](https://codecov.io/gh/tomhosking/torchseq)  ![TorchSeq](https://github.com/tomhosking/torchseq/workflows/TorchSeq/badge.svg) [![Documentation Status](https://readthedocs.org/projects/torchseq/badge/?version=latest)](https://torchseq.readthedocs.io/en/latest/?badge=latest)


TorchSeq is a research-first sequence modelling framework built in Pytorch. It's designed to be easy to hack, without crazy levels of inheritance and easy access to the guts of the model.

## Setup

From a fresh venv, run:

```
pip install -r requirements.txt
pip install -e .
```

```
# Optional: this will download a bunch of pretrained models from HuggingFace
python3 ./scripts/download_models.py
```

Datasets go in `./data/` by default, e.g. `./data/squad/`.

Download our cleaned and clustered split of Paralex [here](http://tomho.sk/models/torchseq/paralex.zip), which goes in `./data/wikianswers-pp/`.


## Quickstart

Have a look in `./examples/` for some notebooks that show how to interact with pretrained models, or try training your own models using the configuration files in `./configs/`:

```
torchseq --train --config ./configs/qg_bart.json
```

Then evaluate it with a snippet like this:

```
import json
from torchseq.agents.aq_agent import AQAgent
from torchseq.datasets.qa_loader import QADataLoader
from torchseq.utils.config import Config
from torchseq.metric_hooks.textual import TextualMetricHook
import torch

model_path = '../models/examples/20210223_191015_qg_bart/'


# Load the config
with open(model_path + 'config.json') as f:
    cfg_dict = json.load(f)

config = Config(cfg_dict)

# Load the model
instance = AQAgent(config=config, run_id=None, output_path="./runs/examples/qg_bert_eval", data_path="../data/", silent=False, verbose=False)
instance.load_checkpoint(model_path + 'model/checkpoint.pt')
instance.model.eval()

# Create a dataset
data_loader = QADataLoader(config,  data_path="../data/")

# Run inference on the test split
test_loss, all_metrics, (pred_output, gold_output, gold_input), memory_values_to_return = instance.inference(data_loader.test_loader, metric_hooks=[TextualMetricHook(config, 'c', 'q')])

# Done!
print(all_metrics['bleu'])
```
> 21.065812894587264



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
> ['Who was the oldest cat?', 'How long did Creme Puff live?']


## Pretrained models

Unzip the models to their own folder within `./models/examples/`, e.g. `./models/examples/20210222_145034_qg_transformer/` - then load the model into torchseq using the flag `--load ./models/examples/20210222_145034_qg_transformer/` or the snippet above.

### Question Generation

A transformer model trained on SQuAD - \[ [Download](http://tomho.sk/models/torchseq/qg_transformer.zip) \]

A transformer model using BERT encodings trained on SQuAD - \[ [Download](http://tomho.sk/models/torchseq/qg_bert.zip) \]

BART fine tuned on SQuAD - \[ [Download](http://tomho.sk/models/torchseq/qg_bart.zip) \]

### Paraphrasing

A vanilla autoencoder trained on Paralex - \[ [Download](http://tomho.sk/models/torchseq/paraphrasing_ae.zip) \]

A VAE model trained on Paralex - \[ [Download](http://tomho.sk/models/torchseq/paraphrasing_vae.zip) \]

A VQ-VAE model trained on Paralex - \[ [Download](http://tomho.sk/models/torchseq/paraphrasing_vqvae.zip) \]

Separator for Paralex and QQP


## Citation

If you find TorchSeq useful, please cite one of my papers!

```
@inproceedings{hosking-lapata-2021-factorising,
    title = "Factorising Meaning and Form for Intent-Preserving Paraphrasing",
    author = "Hosking, Tom  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.112",
    doi = "10.18653/v1/2021.acl-long.112",
    pages = "1405--1418",
}
```
```
@inproceedings{hosking-etal-2022-hierarchical,
    title = "Hierarchical Sketch Induction for Paraphrase Generation",
    author = "Hosking, Tom  and
      Tang, Hao  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.178",
    doi = "10.18653/v1/2022.acl-long.178",
    pages = "2489--2501",
}
```