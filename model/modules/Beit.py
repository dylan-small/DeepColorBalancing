from ..builder import loader
from transformers import AutoModel


class BeitBuilder(loader.Builder):

    def load(self):
        model = AutoModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k').base_model
        return model, model.config.hidden_size
