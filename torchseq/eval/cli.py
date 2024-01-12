import logging
import json
import importlib
import torch
import os

import torchseq
from torchseq.eval.args import parse_eval_args
import torchseq.eval.recipes as recipes


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s", datefmt="%H:%M"
    )
    logger = logging.getLogger("Eval")

    args = parse_eval_args()

    if args.version:
        print(torchseq.__version__)
        return

    if args.amp:
        logger.info("Using matmul precision = high")
        torch.set_float32_matmul_precision("high")

    logger.info("TorchSeq eval runner")

    cfg_patch = {}

    config_path = os.path.join(args.load, "config.json")

    if not os.path.exists(config_path):
        raise Exception("No config file found in path {:}".format(args.load))

    # First load the model
    # instance = model_from_path(args.load, config_patch=cfg_patch, use_cuda=(not args.cpu))
    # logger.info("Loaded model from {:}".format(args.load))

    # Then load the data
    # ???

    # Run the recipe

    recipe_module = importlib.import_module("torchseq.eval.recipes." + args.recipe, None)
    if recipe_module is not None:
        recipe: recipes.EvalRecipe = recipe_module.Recipe(args.load, args.data_path, args.test, args.cpu, logger)
        result = recipe.run()
    else:
        logger.error("No recipe called {:} found!".format(args.recipe))

    # Post-process

    # Publish!

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
