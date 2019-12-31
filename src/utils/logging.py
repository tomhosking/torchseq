from torch.utils.tensorboard import SummaryWriter

from args import FLAGS as FLAGS

writer = None

# TODO: This is all really crappy
def add_to_log(key, value, iteration, run_id):
    global writer
    if writer is None:
        writer = SummaryWriter(FLAGS.output_path + '/'+run_id + '/logs')

    writer.add_scalar(key, value, iteration)