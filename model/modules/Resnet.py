from ..builder import loader
from transformers import AutoModel


class ResnetBuilder(loader.Builder):

    def __init__(self, input_size):
        super().__init__(self, input_size)

    def load(self):
        return AutoModel.from_pretrained('microsoft/resnet-50').backbone
