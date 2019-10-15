class Config(object):
    def __init__(self, d):
        self.data = d
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [Config(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, Config(b) if isinstance(b, dict) else b)