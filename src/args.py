from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "./configs/default.json", "Path to config file")

flags.DEFINE_bool("train", False, "Run training?")
flags.DEFINE_bool("validate", False, "Eval on dev set?")
flags.DEFINE_bool("test", False, "Eval on test set?")
flags.DEFINE_bool("preprocess", False, "Run preprocessing on data")

flags.DEFINE_string("load_chkpt", None, "Path to chkpt file")

# Environment setup
flags.DEFINE_string("data_path", "./data/", "Path to data sources")
flags.DEFINE_bool("cuda", True, "Use GPU?")

flags.DEFINE_integer("seed", 123, "Random seed value")