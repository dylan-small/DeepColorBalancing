from torch import nn


class Model(nn.Module):

    def __init__(self, pretrained_model):
        self.headless = pretrained_model.load()
        self.out = nn.Linear(self.headless.config.hidden_size, 3)

    def forward(self, x):
        x = self.headless(x)['last_hidden_state']
        x = self.out(x)
        return x
