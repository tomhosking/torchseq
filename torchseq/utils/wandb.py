import os

import wandb


def wandb_init(config, run_id=None, path=None):
    if "WANDB_API_KEY" in os.environ and "WANDB_USERNAME" in os.environ and os.environ.get("WANDB_USERNAME", "") != "":
        # W&B hierarchy is: project > group > job_type > name > id
        wandb.init(
            project=config.tag,
            group=config.get("group", None),
            job_type=config.get("job_type", None),
            config=config.data,
            name=config.get("name", None),
            id=run_id,
            dir=path,
        )
    elif os.environ.get("WANDB_MODE", None) == "disabled":
        wandb.init(
            project=config.tag,
            group=config.get("group", None),
            job_type=config.get("job_type", None),
            config=config.data,
            name=config.get("name", None),
            id=run_id,
            dir=path,
            mode="disabled",
        )


def wandb_log(data, step=None):
    if (
        "WANDB_API_KEY" in os.environ and "WANDB_USERNAME" in os.environ and os.environ.get("WANDB_USERNAME", "") != ""
    ) or os.environ.get("WANDB_MODE", None) == "disabled":
        wandb.log(data, step)
