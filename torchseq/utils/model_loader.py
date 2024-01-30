import json
from torchseq.agents.aq_agent import AQAgent
from torchseq.agents.seq2seq_agent import Seq2SeqAgent
from torchseq.agents.retrieval_agent import RetrievalAgent
from torchseq.agents.lm_agent import LangModelAgent
from torchseq.utils.config import Config, merge_cfg_dicts
from torchseq.utils.config_migration import check_config
import torch
import logging


AGENT_TYPES = {
    "aq": AQAgent,
    "langmodel": LangModelAgent,
    "para": Seq2SeqAgent,
    "seq2seq": Seq2SeqAgent,
    "autoencoder": Seq2SeqAgent,
    "exemplarguided": Seq2SeqAgent,
    "retrieval": RetrievalAgent,
}

logger = logging.getLogger("Loader")


def config_from_path(path_to_model, config_patch=None):
    with open(path_to_model + "/config.json") as f:
        cfg_dict = json.load(f)

    if config_patch is not None:
        cfg_dict = merge_cfg_dicts(cfg_dict, config_patch)

    config = Config(cfg_dict)

    return config


def model_from_path(
    path_to_model,
    output_path="./runs/",
    data_path="./data/",
    config_patch=None,
    training_mode=False,
    run_id=None,
    **kwargs,
):
    torch.cuda.empty_cache()

    config = config_from_path(path_to_model, config_patch)

    if check_config(config.data):
        logger.warning("Config is outdated! Run the migration script to update it")

    # run_id = path_to_model.split("/")[-1] if run_id is False else run_id

    checkpoint_path = path_to_model + "/model/checkpoint.pt"

    instance = AGENT_TYPES[config.task](
        config,
        run_id,
        output_path,
        data_path=data_path,
        cache_root=path_to_model,
        training_mode=training_mode,
        **kwargs,
    )
    instance.load_checkpoint(checkpoint_path)
    if not training_mode:
        instance.model.eval()

    return instance
