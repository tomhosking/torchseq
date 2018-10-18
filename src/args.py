from absl import flags

FLAGS = flags.FLAGS

# Environment setup
flags.DEFINE_string("data_path", None, "Path to data sources")
flags.DEFINE_bool("cuda", False, "Use GPU?")

flags.DEFINE_integer("seed", 12345, "Random seed value")


# Preprocessing
flags.DEFINE_integer("crop_sentences_before", 1, "Crop the context to this many sentences before the answer span")
flags.DEFINE_integer("crop_sentences_after", 1, "Crop the context to this many sentences after the answer span")
flags.DEFINE_integer("vocab_size", 2000, "Size of shortlist vocab")


# Training hyperparams
flags.DEFINE_integer("batch_size", 8, "Training bathc size")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")


# Model hyperparams
flags.DEFINE_integer("embedding_dim", 200, "Word embedding dimensionality")
