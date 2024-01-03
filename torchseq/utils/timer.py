from time import perf_counter


class Timer:
    def __init__(self, template="Time: {:.3f} seconds", show_readout=True) -> None:
        self.show_readout = show_readout
        self.template = template

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = self.template.format(self.time)
        if self.show_readout:
            print(self.readout)
