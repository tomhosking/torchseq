# from absl import flags

# FLAGS = flags.FLAGS

# flags.DEFINE_str("config", "./configs/default.json", "Path to config file")
# flags.DEFINE_multi_str("patch", None, "Config mask(s) to overwrite main config with")

# flags.DEFINE_bool("train", False, "Run training?")
# flags.DEFINE_bool("validate", False, "Eval on dev set?")
# flags.DEFINE_bool("validate_train", False, "Eval on training set?")
# flags.DEFINE_bool("test", False, "Eval on test set?")
# flags.DEFINE_bool("preprocess", False, "Run preprocessing on data")
# flags.DEFINE_bool("silent", False, "Hide verbose output (useful when running on scheduler)")

# flags.DEFINE_str("load_chkpt", None, "Path to chkpt file")
# flags.DEFINE_str("load", None, "Path to model")

# # Environment setup
# flags.DEFINE_str("data_path", "./data/", "Path to data sources")
# flags.DEFINE_str("output_path", "./runs/", "Path to output folder")
# # flags.DEFINE_bool("cuda", True, "Use GPU?")

# flags.DEFINE_integer("seed", 123, "Random seed value")

import argparse

parser = argparse.ArgumentParser(
    description="TorchSeq",
)

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
parser.add_argument("--preprocess", action="store_true", help="Run prepro?")
parser.add_argument("--silent", action="store_true", help="Disable logging")

# Model loading
parser.add_argument("--load_chkpt", type=str, metavar="CHECKPOINT", default=None, help="Path to checkpoint file")
parser.add_argument("-l", "--load", type=str, metavar="MODEL", default=None, help="Path to model folder")

# Paths
parser.add_argument("-d", "--data_path", type=str, metavar="DATA", default="./data/", help="Path to data sources")
parser.add_argument("-o", "--output_path", type=str, metavar="OUTPUT", default="./runs/", help="Path to output folder")

# Runtime
parser.add_argument("--cpu", action="store_true", help="Disable CUDA")


args = parser.parse_args()
