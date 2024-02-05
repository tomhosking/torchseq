from genericpath import isdir


def migrate_optimizers_23(cfg_dict, check_only=False):
    # Migrate old style optimizer definition
    if "optimizer" not in cfg_dict["training"]:
        if check_only:
            return True
        optimizer_cfg_dict = {
            "type": cfg_dict["training"].get("opt", "adam"),
            "lr": cfg_dict["training"].get("lr", 1e-4),
            "beta1": cfg_dict["training"].get("beta1", 0.9),
            "beta2": cfg_dict["training"].get("beta2", 0.98),
            "lr_schedule": cfg_dict["training"].get("lr_schedule", False),
            "lr_warmup_steps": cfg_dict["training"].get("lr_warmup_steps", 10000),
        }
        cfg_dict["training"].pop("lr")
        cfg_dict["training"].pop("opt")
        if "beta1" in cfg_dict["training"]:
            cfg_dict["training"].pop("beta1")
        if "beta2" in cfg_dict["training"]:
            cfg_dict["training"].pop("beta2")
        if "lr_schedule" in cfg_dict["training"]:
            cfg_dict["training"].pop("lr_schedule")
        if "lr_warmup_steps" in cfg_dict["training"]:
            cfg_dict["training"].pop("lr_warmup_steps")

        cfg_dict["training"]["optimizer"] = optimizer_cfg_dict

    return False if check_only else cfg_dict


def migrate_23_to_24_encdec(cfg_dict, check_only=False):
    # Migrate old style encdec configs
    if "encdec" in cfg_dict:
        if check_only:
            return True

        cfg_dict["encoder"] = {**cfg_dict["encdec"], **cfg_dict["encoder"]}
        cfg_dict["decoder"] = {**cfg_dict["encdec"], **cfg_dict["encoder"]}
        cfg_dict["encoder"]["num_layers"] = cfg_dict["encoder"]["num_encoder_layers"]
        cfg_dict["decoder"]["num_layers"] = cfg_dict["decoder"]["num_decoder_layers"]
        cfg_dict.pop("encdec")
        cfg_dict["encoder"].pop("num_encoder_layers")
        cfg_dict["encoder"].pop("num_decoder_layers")
        cfg_dict["decoder"].pop("num_encoder_layers")
        cfg_dict["decoder"].pop("num_decoder_layers")

    return False if check_only else cfg_dict


def migrate_selfret_to_hero(cfg_dict, check_only=False):
    if "self_retrieval" in cfg_dict["eval"]["metrics"]:
        if check_only:
            return True
        cfg_dict["eval"]["metrics"]["opsumm_cluster_aug"] = cfg_dict["eval"]["metrics"]["self_retrieval"]
        cfg_dict["eval"]["metrics"].pop("self_retrieval")
    return False if check_only else cfg_dict


all_migrations = [migrate_optimizers_23, migrate_23_to_24_encdec, migrate_selfret_to_hero]


def check_config(config):
    would_modify = False
    for migration in all_migrations:
        would_modify = would_modify or migration(config, check_only=True)
    return would_modify


def migrate_config(config):
    for migration in all_migrations:
        config = migration(config, check_only=False)
    return config


if __name__ == "__main__":
    import argparse
    import json
    import os

    def migrate_and_save(path):
        with open(path) as f:
            cfg_dict = json.load(f)

        with open(path.replace("config.json", "config.bak.json"), "w") as f:
            json.dump(cfg_dict, f, indent=4)

        cfg_dict = migrate_config(cfg_dict)

        with open(path, "w") as f:
            json.dump(cfg_dict, f, indent=4)

    parser = argparse.ArgumentParser(
        description="TorchSeq",
    )

    parser.add_argument("path", type=str, metavar="FILE", default=None, help="Path to config file")

    args = parser.parse_args()

    if os.path.isdir(args.path):
        filepaths = [
            os.path.join(args.path, f) for f in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, f))
        ]
        for path in filepaths:
            if path[-11:] == "config.json":
                print("Migrating {:}".format(path))
                migrate_and_save(path)

    else:
        migrate_and_save(args.path)
