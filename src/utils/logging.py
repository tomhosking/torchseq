from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def add_to_log(key, value, iteration):
    

    writer.add_scalar(key, value, iteration)