class Config(object):
    def __init__(self, d):
        self.data = d
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Config(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Config(b) if isinstance(b, dict) else b)


def merge_cfg_dicts(main_cfg, cfg_mask):
    for k, v in cfg_mask.items():
        if k in main_cfg and isinstance(main_cfg[k], dict) and isinstance(cfg_mask[k], dict):
            merge_cfg_dicts(main_cfg[k], cfg_mask[k])
        else:
            main_cfg[k] = cfg_mask[k]

    return main_cfg
