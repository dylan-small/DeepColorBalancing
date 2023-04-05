from ..builder import loader
from transformers import AutoModel


class ResnetBuilder(loader.Builder):

    def load(self):
        model = AutoModel.from_pretrained('microsoft/resnet-50').base_model
        return model, model.config.hidden_sizes[-1]
