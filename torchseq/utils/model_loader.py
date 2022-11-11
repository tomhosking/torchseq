import json
from torchseq.agents.aq_agent import AQAgent
from torchseq.agents.seq2seq_agent import Seq2SeqAgent
from torchseq.agents.lm_agent import LangModelAgent
from torchseq.agents.meta_learning_agent import MetaLearningAgent
from torchseq.agents.exemplar_agent import ExemplarGuidedAgent
from torchseq.utils.config import Config, merge_cfg_dicts
import torch


AGENT_TYPES = {
    "aq": AQAgent,
    "langmodel": LangModelAgent,
    "para": Seq2SeqAgent,
    "seq2seq": Seq2SeqAgent,
    "autoencoder": Seq2SeqAgent,
    "metalearning": MetaLearningAgent,
    "exemplarguided": ExemplarGuidedAgent,
}


def model_from_path(
    path_to_model, output_path="./runs/", data_path="./data/", config_patch=None, training_mode=False, **kwargs
):
    torch.cuda.empty_cache()

    with open(path_to_model + "/config.json") as f:
        cfg_dict = json.load(f)

    if config_patch is not None:
        cfg_dict = merge_cfg_dicts(cfg_dict, config_patch)

    run_id = path_to_model.split("/")[-1]

    config = Config(cfg_dict)

    checkpoint_path = path_to_model + "/model/checkpoint.pt"

    instance = AGENT_TYPES[config.task](
        config, run_id, output_path, data_path=data_path, cache_root=path_to_model, **kwargs
    )
    instance.load_checkpoint(checkpoint_path)
    if not training_mode:
        instance.model.eval()

    return instance
