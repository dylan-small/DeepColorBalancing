from ..builder import loader
from transformers import AutoModel


class ViTBuilder(loader.Builder):

    def load(self):
        model = AutoModel.from_pretrained('google/vit-base-patch16-224').base_model
        return model, model.config.hidden_size
