import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="TorchSeq",
    )

    parser.add_argument("-V", "--version", action="store_true", help="Display version")

    # Config stuff
    parser.add_argument(
        "-c", "--config", type=str, metavar="CONFIG", default="./configs/default.json", help="Path to config file"
    )
    parser.add_argument(
        "-p",
        "--patch",
        type=str,
        metavar="PATCH",
        default=None,
        help="Config mask(s) to overwrite main config with",
        action="append",
    )

    # Actions
    parser.add_argument("--train", action="store_true", help="Run training?")
    parser.add_argument("--validate", action="store_true", help="Run eval on dev?")
    parser.add_argument("--validate_train", action="store_true", help="Run eval on train?")
    parser.add_argument("--test", action="store_true", help="Run eval on test?")
    parser.add_argument("--silent", action="store_true", help="Disable logging")
    parser.add_argument("--verbose", action="store_true", help="Extra logging")
    parser.add_argument(
        "--reload_after_train", action="store_true", help="Reload model after training to do a validation run"
    )
    parser.add_argument(
        "--copy_chkpt",
        action="store_true",
        help="Save a copy of the checkpoint in current output dir, even if loading from elsewhere",
    )

    # Model loading
    parser.add_argument("--load_chkpt", type=str, metavar="CHECKPOINT", default=None, help="Path to checkpoint file")
    parser.add_argument("-l", "--load", type=str, metavar="MODEL", default=None, help="Path to model folder")
    parser.add_argument("--nocache", action="store_true", help="Disable loading from an old cache")

    # Paths
    parser.add_argument("-d", "--data_path", type=str, metavar="DATA", default="./data/", help="Path to data sources")
    parser.add_argument(
        "-o", "--output_path", type=str, metavar="OUTPUT", default="./runs/", help="Path to output folder"
    )

    # Runtime
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    return args
