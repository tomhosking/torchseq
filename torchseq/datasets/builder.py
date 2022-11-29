from torchseq.datasets.paraphrase_loader import ParaphraseDataLoader
from torchseq.datasets.qa_loader import QADataLoader
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.datasets.lm_loader import LangmodellingDataLoader


def dataloader_from_config(config, data_path="./data", train_samples=None, dev_samples=None, test_samples=None):
    # define data_loader
    if config.training.dataset is None:
        data_loader = None
    elif (
        config.training.dataset
        in [
            "paranmt",
            "parabank",
            "kaggle",
            "parabank-qs",
            "para-squad",
            "models/squad-udep",
            "models/squad-constituency",
            "models/squad-udep-deptree",
            "models/qdmr-squad",
            "models/nq_newsqa-udep",
            "models/nq_newsqa-udep-deptree",
            "models/squad_nq_newsqa-udep",
            "models/squad_nq_newsqa-udep-deptree",
            "models/naturalquestions-udep",
            "models/newsqa-udep",
            "models/naturalquestions-udep-deptree",
            "models/newsqa-udep-deptree",
        ]
        or config.training.dataset[:5] == "qdmr-"
        or "kaggle-" in config.training.dataset
    ):
        data_loader = ParaphraseDataLoader(
            config=config,
            data_path=data_path,
            train_samples=train_samples,
            dev_samples=dev_samples,
            test_samples=test_samples,
        )
    elif (
        config.training.dataset
        in [
            "squad",
            "newsqa",
            "msmarco",
            "naturalquestions",
            "drop",
            "nq_newsqa",
            "squad_nq_newsqa",
            "inquisitive",
        ]
        or config.training.dataset[:5] == "squad"
        or config.training.dataset[:3] == "qa/"
    ):
        data_loader = QADataLoader(
            config=config,
            data_path=data_path,
            train_samples=train_samples,
            dev_samples=dev_samples,
            test_samples=test_samples,
        )
    elif config.training.dataset in [
        "json",
    ]:
        data_loader = JsonDataLoader(
            config=config,
            data_path=data_path,
            train_samples=train_samples,
            dev_samples=dev_samples,
            test_samples=test_samples,
        )
    elif config.training.dataset in ["ptb", "wikitext103"]:
        data_loader = LangmodellingDataLoader(
            config=config,
            data_path=data_path,
            train_samples=train_samples,
            dev_samples=dev_samples,
            test_samples=test_samples,
        )
    else:
        raise Exception("Unrecognised dataset: {:}".format(config.training.dataset))

    return data_loader
