from ..builder import loader
from transformers import AutoModel


class BeitBuilder(loader.Builder):

    def __init__(self, input_size):
        super().__init__(self, input_size)

    def load(self):
        return AutoModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k').backbone
