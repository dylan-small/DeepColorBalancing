from ..builder import loader
from transformers import AutoModel


class ViTBuilder(loader.Builder):

    def load(self):
        return AutoModel.from_pretrained('google/vit-base-patch16-224').backbone
