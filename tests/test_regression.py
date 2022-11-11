import json
import os
from datetime import datetime

import torch


from . import utils as test_utils

from torchseq.agents.aq_agent import AQAgent
from torchseq.agents.seq2seq_agent import Seq2SeqAgent
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config
from torchseq.utils.seed import set_seed

from torchseq.datasets.builder import dataloader_from_config


@test_utils.slow
def test_bert_embeds():

    CONFIG = "./models/examples/20210222_145021_qg_bert/config.json"
    CHKPT = "./models/examples/20210222_145021_qg_bert/model/checkpoint.pt"
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./runs/"
    SEED = 123

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()

    assert use_cuda, "This test needs to run on GPU!"

    with open(CONFIG) as f:
        cfg_dict = json.load(f)

        cfg_dict["eval"]["truncate_dataset"] = 100

        config = Config(cfg_dict)

    set_seed(SEED)

    if config.task == "aq":
        agent = AQAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)
    elif config.task in ["para", "autoencoder"]:
        agent = Seq2SeqAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)

    data_loader = dataloader_from_config(config, DATA_PATH)

    agent.load_checkpoint(CHKPT)
    agent.model.eval()
    loss, metrics, output, memory = agent.validate(data_loader, force_save_output=True, save_model=False)

    # Now check the output (for first 100 samples)
    assert abs(loss.item() - 2.94483) < 1e-3, "Loss is different to expected!"
    assert "bleu" in metrics, "BLEU is missing from output metrics!"
    assert abs(metrics["bleu"] - 18.47254) < 1e-2, "BLEU score is different to expected!"

    # Targets for full dataset:
    # assert abs(loss.item() - 2.833) < 1e-3, "Loss is different to expected!"
    # assert "bleu" in metrics, "BLEU is missing from output metrics!"
    # assert abs(metrics["bleu"] - 17.22) < 1e-2, "BLEU score is different to expected!"


@test_utils.slow
def test_paraphrasing_vae():

    CONFIG = "./models/examples/20210222_152157_paraphrasing_vae/config.json"
    CHKPT = "./models/examples/20210222_152157_paraphrasing_vae/model/checkpoint.pt"
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./runs/"
    SEED = 123

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()

    assert use_cuda, "This test needs to run on GPU!"

    with open(CONFIG) as f:
        cfg_dict = json.load(f)

        cfg_dict["eval"]["truncate_dataset"] = 100
        cfg_dict["eval"]["vae_use_map"] = True

        config = Config(cfg_dict)

    set_seed(SEED)

    if config.task == "aq":
        agent = AQAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)
    elif config.task in ["para", "autoencoder"]:
        agent = Seq2SeqAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)

    data_loader = dataloader_from_config(config, DATA_PATH)

    agent.load_checkpoint(CHKPT)
    agent.model.eval()
    loss, metrics, output, memory = agent.validate(data_loader, force_save_output=True, save_model=False)

    # Now check the output (for first 100 samples)
    assert abs(loss.item() - 2.22916) < 1e-3, "Loss is different to expected!"
    assert "bleu" in metrics, "BLEU is missing from output metrics!"
    assert abs(metrics["bleu"] - 36.18769) < 1e-2, "BLEU score is different to expected!"

    # Targets for full dataset:
    # assert abs(loss.item() - ???) < 1e-3, "Loss is different to expected!"
    # assert "bleu" in metrics, "BLEU is missing from output metrics!"
    # assert abs(metrics["bleu"] - ???) < 1e-2, "BLEU score is different to expected!"


@test_utils.slow
def test_qg_transformer():

    CONFIG = "./models/examples/20210222_145034_qg_transformer/config.json"
    CHKPT = "./models/examples/20210222_145034_qg_transformer/model/checkpoint.pt"
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./runs/"
    SEED = 123

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()

    assert use_cuda, "This test needs to run on GPU!"

    with open(CONFIG) as f:
        cfg_dict = json.load(f)
        cfg_dict["eval"]["truncate_dataset"] = 100

        config = Config(cfg_dict)

    set_seed(SEED)

    if config.task == "aq":
        agent = AQAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)
    elif config.task in ["para", "autoencoder"]:
        agent = Seq2SeqAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)

    data_loader = dataloader_from_config(config, DATA_PATH)

    agent.load_checkpoint(CHKPT)
    agent.model.eval()
    loss, metrics, output, memory = agent.validate(data_loader, force_save_output=True, save_model=False)

    # Now check the output (for first 100 samples)
    assert abs(loss.item() - 3.16849) < 1e-3, "Loss is different to expected!"
    assert "bleu" in metrics, "BLEU is missing from output metrics!"
    assert abs(metrics["bleu"] - 18.38744) < 1e-2, "BLEU score is different to expected!"


@test_utils.slow
def test_qg_bart():

    CONFIG = "./models/examples/20210223_191015_qg_bart/config.json"
    CHKPT = "./models/examples/20210223_191015_qg_bart/model/checkpoint.pt"
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./runs/"
    SEED = 123

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()

    assert use_cuda, "This test needs to run on GPU!"

    with open(CONFIG) as f:
        cfg_dict = json.load(f)

        cfg_dict["eval"]["truncate_dataset"] = 100

        config = Config(cfg_dict)

    set_seed(SEED)

    if config.task == "aq":
        agent = AQAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)
    elif config.task in ["para", "autoencoder"]:
        agent = Seq2SeqAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)

    data_loader = dataloader_from_config(config, DATA_PATH)

    agent.load_checkpoint(CHKPT)
    loss, metrics, output, memory = agent.validate(data_loader, force_save_output=True, save_model=False)

    # Now check the output (for first 100 samples)
    assert abs(loss.item() - 1.23542) < 1e-3, "Loss is different to expected!"  #
    assert "bleu" in metrics, "BLEU is missing from output metrics!"
    assert abs(metrics["bleu"] - 25.918) < 1e-2, "BLEU score is different to expected!"


@test_utils.slow
def test_separator():

    CONFIG = "./models/examples/separator-wa/config.json"
    CHKPT = "./models/examples/separator-wa/model/checkpoint.pt"
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./runs/"
    SEED = 123

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()

    assert use_cuda, "This test needs to run on GPU!"

    examples = [
        {
            "sem_input": "What is the weight of an average moose?",
            "syn_input": "How much is a surgeon's income?",
            "tgt": "how much is a moose's weight?",
        }
    ]

    with open(CONFIG) as f:
        cfg_dict = json.load(f)

        cfg_dict["eval"]["truncate_dataset"] = 100
        cfg_dict["eval"]["vae_use_map"] = True
        cfg_dict["beam_search"] = {"beam_width": 4, "beam_expansion": 2, "length_alpha": 1.0}

        config = Config(cfg_dict)

    set_seed(SEED)

    if config.task == "aq":
        agent = AQAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)
    elif config.task in ["para", "autoencoder"]:
        agent = Seq2SeqAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)

    data_loader = JsonDataLoader(config, DATA_PATH, test_samples=examples)

    agent.load_checkpoint(CHKPT)
    agent.model.eval()
    loss, metrics, (pred_output, gold_output, gold_input), memory = agent.inference(data_loader.test_loader)

    # Now check the output (for first 100 samples)
    assert pred_output[0] == examples[0]["tgt"]


@test_utils.slow
def test_hrq():

    CONFIG = "./models/examples/hrqvae_v2/20220109_182050_wa_base_jointtrained_TEST/config_usehrq.json"
    CHKPT = "./models/examples/hrqvae_v2/20220109_182050_wa_base_jointtrained_TEST/model/checkpoint.pt"
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./runs/"
    SEED = 123

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()

    assert use_cuda, "This test needs to run on GPU!"

    examples = [
        {
            "sem_input": "What is the weight of an average moose?",
            "syn_input": "How much does a surgeon make?",
            "tgt": "how much does a moose weight?",
        },
        {
            "sem_input": "What is the weight of an average moose?",
            "syn_input": "What is the income of a surgeon?",
            "tgt": "what is the weight of a moose?",
        },
    ]

    with open(CONFIG) as f:
        cfg_dict = json.load(f)

        cfg_dict["eval"]["truncate_dataset"] = 100
        cfg_dict["eval"]["vae_use_map"] = True
        cfg_dict["bottleneck"]["code_predictor"]["infer_codes"] = False
        cfg_dict["beam_search"] = {"beam_width": 4, "beam_expansion": 2, "length_alpha": 1.0}

        config = Config(cfg_dict)

    set_seed(SEED)

    if config.task == "aq":
        agent = AQAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)
    elif config.task in ["para", "autoencoder"]:
        agent = Seq2SeqAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)

    data_loader = JsonDataLoader(config, DATA_PATH, test_samples=examples)

    agent.load_checkpoint(CHKPT)
    agent.model.eval()
    loss, metrics, (pred_output, gold_output, gold_input), memory = agent.inference(data_loader.test_loader)

    # Now check the output (for first 100 samples)
    assert pred_output[0] == examples[0]["tgt"]
    assert pred_output[1] == examples[1]["tgt"]


@test_utils.slow
def test_hrq_codepred():

    CONFIG = "./models/examples/hrqvae_v2/20220109_182050_wa_base_jointtrained_TEST/config_usehrq.json"
    CHKPT = "./models/examples/hrqvae_v2/20220109_182050_wa_base_jointtrained_TEST/model/checkpoint.pt"
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./runs/"
    SEED = 123

    # Most of this is copied from main.py
    use_cuda = torch.cuda.is_available()

    assert use_cuda, "This test needs to run on GPU!"

    examples = [
        {
            "sem_input": "What is the weight of an average moose?",
            "syn_input": "What is the income of a surgeon?",
            "tgt": "how much does a moose weight?",
        }
    ]

    with open(CONFIG) as f:
        cfg_dict = json.load(f)

        cfg_dict["eval"]["truncate_dataset"] = 100
        cfg_dict["eval"]["vae_use_map"] = True
        cfg_dict["bottleneck"]["code_predictor"]["infer_codes"] = True
        cfg_dict["beam_search"] = {"beam_width": 4, "beam_expansion": 2, "length_alpha": 1.0}

        config = Config(cfg_dict)

    set_seed(SEED)

    if config.task == "aq":
        agent = AQAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)
    elif config.task in ["para", "autoencoder"]:
        agent = Seq2SeqAgent(config, None, OUTPUT_PATH, DATA_PATH, silent=True, training_mode=False)

    data_loader = JsonDataLoader(config, DATA_PATH, test_samples=examples)

    agent.load_checkpoint(CHKPT)
    agent.model.eval()
    loss, metrics, (pred_output, gold_output, gold_input), memory = agent.inference(data_loader.test_loader)

    # Now check the output (for first 100 samples)
    assert pred_output[0] == examples[0]["tgt"]
