
class Builder(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def load(self):
        raise NotImplementedError()
