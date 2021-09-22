import os
import shutil
import torch
import numpy as np


class Cache:
    def __init__(self, output_path=None):

        self.path = os.path.join(output_path, "cache")
        os.makedirs(self.path, exist_ok=True)

    def load(self, key):
        if not os.path.exists(os.path.join(self.path, f"{key}.pt")):
            return None
        # obj = np.load(os.path.join(self.path, f"{key}.npy"))
        # if isinstance(obj, np.ndarry):
        #     return torch.from_numpy(obj)
        # else:
        #     return obj
        return torch.load(os.path.join(self.path, f"{key}.pt"))

    def save(self, key, obj):
        # np.save(os.path.join(self.path, f"{key}.npy"), obj)
        torch.save(obj, os.path.join(self.path, f"{key}.pt"))
