from model.pretrained.ViT import ViTBuilder
from model.MainModel import Model

vit = ViTBuilder((224, 224))
model = Model(vit)

