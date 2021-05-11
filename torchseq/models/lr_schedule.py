from torch.optim.lr_scheduler import LambdaLR


# TODO: control diff types of schedule in a more granular way - eg we may want linear warmup -> poly/exp decay
def get_lr(base_lr, step, scheduled=False, warmup=True):

    if scheduled:
        step = max(step, 1)
        warmup_steps = 10000
        if warmup:
            return base_lr * min(pow(step, -0.5), step * pow(warmup_steps, -1.5))
        else:
            return base_lr * min(pow(step, -0.5), warmup_steps * pow(warmup_steps, -1.5))
    else:
        return base_lr


def get_scheduler(optimizer, base_lr, scheduled=False, warmup=True, num_warmup_steps=10000, last_epoch=-1):
    def lr_lambda(current_step: int):
        if scheduled:
            step = max(current_step, 1)
            if warmup:
                return min(pow(step, -0.5), step * pow(num_warmup_steps, -1.5))
            else:
                return min(pow(step, -0.5), num_warmup_steps * pow(num_warmup_steps, -1.5))
        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)
