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
        if "lr_warmup_steps" in cfg_dict["training"]:
            cfg_dict["training"].pop("beta1")
        if "lr_warmup_steps" in cfg_dict["training"]:
            cfg_dict["training"].pop("beta2")
        if "lr_warmup_steps" in cfg_dict["training"]:
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

    return False if check_only else cfg_dict


all_migrations = [
    migrate_optimizers_23,
    # migrate_23_to_24_encdec
]


def check_config(config):
    would_modify = False
    for migration in all_migrations:
        would_modify = would_modify or migration(config, check_only=True)
    return would_modify
