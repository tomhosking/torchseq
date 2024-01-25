import argparse


def parse_eval_args(arg_str=None):
    parser = argparse.ArgumentParser(
        description="TorchSeq Evaluation runner",
    )

    parser.add_argument("-V", "--version", action="store_true", help="Display version")

    parser.add_argument("--model", type=str, metavar="MODEL", help="Path to model folder", required=True)

    parser.add_argument("--recipe", type=str, metavar="RECIPE", help="Name of recipe to run", required=True)

    parser.add_argument("--test", action="store_true", help="Use test set")

    # Paths
    parser.add_argument("--data_path", type=str, metavar="DATA", default="./data/", help="Path to data sources")
    parser.add_argument(
        "--output_path", type=str, metavar="OUTPUT", default="./evalruns/", help="Path to output folder"
    )

    # Runtime
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA")
    parser.add_argument("--amp", action="store_true", help="Enable AMP")

    args = parser.parse_args(arg_str)

    return args
