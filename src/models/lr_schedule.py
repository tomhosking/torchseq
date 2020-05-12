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
