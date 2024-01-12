from typing import Dict, Any

from torchseq.utils.model_loader import config_from_path
from torchseq.utils.config import Config
from abc import ABC, abstractmethod


class EvalRecipe(ABC):
    model_path: str
    data_path: str
    config: Config
    split_str: str
    test: bool
    cpu: bool

    def __init__(self, model_path: str, data_path: str, test=False, cpu: bool = False):
        # self.args = args
        self.model_path = model_path
        self.data_path = data_path
        self.config = config_from_path(model_path)
        self.test = test
        self.split_str = "test" if test else "dev"
        self.cpu = cpu

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        raise Exception("Tried to call run() on a base EvalRecipe object!")
