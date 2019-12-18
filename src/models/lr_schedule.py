

def get_lr(base_lr, step, scheduled=False):

        if scheduled:
            step = max(step, 1)
            warmup_steps = 10000
            return base_lr * min(pow(step, -0.5), step * pow(warmup_steps, -1.5))
        else:
            return base_lr