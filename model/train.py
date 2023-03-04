from MainModel import Model
from pretrained import Beit, CustomModel, Resnet, ViT
import torch
from torch import nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, data_loader, model, optimizer, criterion):

    losses = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        losses.update(loss, out.shape[0])

        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), loss=losses))


def validate(epoch, test_loader, model, criterion):

    losses = AverageMeter()
    for idx, (data, target) in enumerate(test_loader):
        if torch.cuda_is_available():
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

        losses.update(loss, out.shape[0])

        if idx % 10 == 0:
            print(('Validation\n: Epoch: [{0}][{1}/{2}]\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                  .format(epoch, idx, len(test_loader), loss=losses))


def main():
    batch_size = 32
    model_name = 'ViT'
    lr = 0.001
    momentum = 0.9
    reg = 0.0001
    epochs = 2

    if model_name == 'ViT':
        pretrained_model = ViT.ViTBuilder((224, 224))
    elif model_name == 'Beit':
        pretrained_model = Beit.BeitBuilder((224, 224))
    elif model_name == 'Resnet':
        pretrained_model = Resnet.ResnetBuilder((224, 224))

    model = Model(pretrained_model)

    print(model)
    if torch.cuda.is_available():
        model = model.cuda()

    train_loader = DataLoader(input_size=model.input_size,dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(input_size=model.input_size, dataset=test_dataset, batch_size=100, shuffle=False)

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=reg)

    for epoch in range(epochs):

        # train loop
        train(epoch, train_loader, model, optimizer, criterion)

        # validation loop
        validate(epoch, test_loader, model, criterion)

    torch.save(model.state_dict(), './model_weights/' + model_name + '.pth')
