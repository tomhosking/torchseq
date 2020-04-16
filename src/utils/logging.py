from torch.utils.tensorboard import SummaryWriter


writer = None

# TODO: This is all really crappy
def add_to_log(key, value, iteration, run_id, output_path):
    global writer
    if writer is None:
        writer = SummaryWriter(output_path + "/" + run_id + "/logs")

    writer.add_scalar(key, value, iteration)
