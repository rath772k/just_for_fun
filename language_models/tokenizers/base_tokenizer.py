class BaseTokenizer:
    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError
