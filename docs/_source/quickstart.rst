Getting Started
===============


Installation
------------

From a fresh ``venv``, run:
```
pip install -r requirements.txt
pip install -e .

python3 -m nltk.downloader punkt
python3 ./scripts/download_models.py
```

Done!

Overview
--------

TorchSeq is a framework for training and evaluating Seq2Seq models, built in PyTorch.




CLI Usage
---------

TorchSeq installs a CLI - to load a model and evaluate it on the test set, run ``torchseq --test --load /path/to/model``.

The CLI options are:

--train     Run training
--validate  Run validation (ie, eval on the dev set)
--validate_train  Run eval on the training set
--test      Run eval on the test set
--silent    Disable verbose output
--reload_after_train    Use in conjunction with one of the eval commands to reload the best checkpoint once training completes, and evaluate using that
--load_chkpt /path/to/checkpoint.pt    Path to checkpoint file
--data_path /path/to/data/   Path to folder containing datasets
--output_path /path/to/output/  Path to dump output
--config,-c /path/to/config.json Path to config file
--patch,-p  /path/to/patch.json Path to 'patch' file(s)
--load  /path/to/model/  Path to a full model (checkpoint + config)
--cpu   Run on CPU
--debug Enable some extra debugging


Scripting
---------

You can also invoke TorchSeq from within a script, like this:

```
from torchseq.agents.para_agent import ParaphraseAgent
from torchseq.datasets.json_loader import JsonDataLoader

from torchseq.utils.config import Config

with open(path_to_model + "/config.json") as f:
    cfg_dict = json.load(f)

config = Config(cfg_dict)

data_loader = JsonDataLoader(config)

checkpoint_path = path_to_model + "/model/checkpoint.pt"

instance = ParaphraseAgent(config=config, run_id=None, output_path="./runs/demo/", silent=True, verbose=False)

instance.load_checkpoint(checkpoint_path)
instance.model.eval()

loss, metrics, output, memory = instance.validate(data_loader, save_model=False)
```

In general, a :torchseq.agents.model_agent.ModelAgent: object is the main controller - once you've created one from a :torchseq.utils.config.Config:, you can train it with :torchseq.agents.base.train: