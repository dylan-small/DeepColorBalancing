from torch import nn


class PretrainedModel(nn.Module):

    def __init__(self, pretrained_builder):
        self.input_size = pretrained_builder.input_size
        self.headless = pretrained_builder.load()
        self.out = nn.Linear(self.headless.config.hidden_size, 3)

    def forward(self, x):
        x = self.headless(x)['last_hidden_state']
        x = self.out(x)
        return x


class CustomModel(nn.Module):
    def __init__(self, custom_builder):
        super(CustomModel, self).__init__()
        self.input_size = custom_builder.input_size
        self.model = custom_builder.load()

    def forward(self, x):
        x = self.model(x)
        return x


