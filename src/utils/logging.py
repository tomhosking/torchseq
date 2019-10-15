from torch.utils.tensorboard import SummaryWriter

writer = None

def add_to_log(key, value, iteration, run_id):
    global writer
    if writer is None:
        writer = SummaryWriter('runs/'+run_id + '/logs')

    writer.add_scalar(key, value, iteration)