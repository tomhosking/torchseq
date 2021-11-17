class OptimizerGroup:
    def __init__(self, optimizer_list):
        self.optimizers = optimizer_list

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def load_state_dict(self, state_dict_arr):
        if not isinstance(state_dict_arr, list):
            state_dict_arr = [state_dict_arr]
        for ix, opt in enumerate(self.optimizers):
            opt.load_state_dict(state_dict_arr[ix])

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def __iter__(self):
        for opt in self.optimizers:
            yield opt

    def __getitem__(self, ix):
        return self.optimizers[ix]

    def __len__(self):
        return len(self.optimizers)


class SchedulerGroup:
    def __init__(self, scheduler_list):
        self.schedulers = scheduler_list

    def step(self):
        for sched in self.schedulers:
            sched.step()

    def load_state_dict(self, state_dict_arr):
        if not isinstance(state_dict_arr, list):
            state_dict_arr = [state_dict_arr]
        for ix, sched in enumerate(self.schedulers):
            sched.load_state_dict(state_dict_arr[ix])

    def state_dict(self):
        return [sched.state_dict() for sched in self.schedulers]

    def __iter__(self):
        for sched in self.schedulers:
            yield sched

    def __getitem__(self, ix):
        return self.schedulers[ix]

    def __len__(self):
        return len(self.schedulers)
