from torch.optim.lr_scheduler import LambdaLR
from math import exp, tanh


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


def get_scheduler(
    optimizer, base_lr, scheduled=False, warmup=True, num_warmup_steps=10000, last_epoch=-1, legacy=True
):
    if legacy:

        def lr_lambda(current_step: int):
            if scheduled:
                step = max(current_step, 1)
                if warmup:
                    return min(pow(step, -0.5), step * pow(num_warmup_steps, -1.5))
                else:
                    return min(pow(step, -0.5), num_warmup_steps * pow(num_warmup_steps, -1.5))
            else:
                return 1.0

    else:
        # Replicate the original BERT LR schedule, but such that it peaks at 1.0
        def lr_lambda(current_step: int):
            if scheduled:
                step = max(current_step, 1)
                if warmup and step <= num_warmup_steps:
                    return min(float(step) / float(num_warmup_steps), 1.0)
                else:
                    return pow(step, -0.5) / pow(num_warmup_steps, -0.5)
            else:
                return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_hyperbolic_schedule(gamma, step):
    return 2 / (1 + exp(-float(step) / float(gamma))) - 1


def get_tanh_schedule(gamma, step):
    return tanh(float(step) / float(gamma)) ** 4
