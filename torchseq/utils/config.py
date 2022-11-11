# A simple class to convert a (nested) dictionary to an object


class Config(object):
    def __init__(self, d):
        self.data = d
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Config(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Config(b) if isinstance(b, dict) else b)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def get_first(self, keys):
        for key in keys:
            if key in self.data:
                return self.data[key]
        else:
            raise KeyError

    def get_path(self, path, default=None):
        if path[0] in self.data:
            if len(path) > 1:
                return getattr(self, path[0]).get_path(path[1:], default)
            else:
                return self.data[path[0]]
        else:
            return default


def merge_cfg_dicts(main_cfg, cfg_mask):
    for k, v in cfg_mask.items():
        if k in main_cfg and isinstance(main_cfg[k], dict) and isinstance(cfg_mask[k], dict):
            merge_cfg_dicts(main_cfg[k], cfg_mask[k])
        elif k == "name" and isinstance(cfg_mask[k], str) and "%" in cfg_mask[k]:
            main_cfg[k] = cfg_mask[k].replace("%", main_cfg[k])
        else:
            main_cfg[k] = cfg_mask[k]

    return main_cfg
