from .MainModel import PretrainedModel, CustomModel
from .modules import Beit, CustomCNN, Resnet, ViT
from loader.Dataloader import ImageDataLoader
import torch
import os
from torch import nn
from datetime import datetime
import matplotlib.pyplot as plt


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

    l = []

    for idx, (data, target) in enumerate(data_loader):

        data = data.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        l.append(float(loss))
        loss.backward()
        optimizer.step()

        losses.update(loss, out.shape[0])

        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), loss=losses))

    return l

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
    model_name = 'Custom'
    lr = 0.001
    reg = 0.0001
    epochs = 2

    if model_name == 'ViT':
        model = PretrainedModel(ViT.ViTBuilder((224, 224)))
    elif model_name == 'Beit':
        model = PretrainedModel(Beit.BeitBuilder((224, 224)))
    elif model_name == 'Resnet':
        model = PretrainedModel(Resnet.ResnetBuilder((224, 224)))
    elif model_name == 'Custom':
        model = CustomModel(CustomCNN.CustomCNNBuilder((256, 256)))

    print(model)
    if torch.cuda.is_available():
        model = model.cuda()

    train_loader = ImageDataLoader(model.input_size, batch_size=8)

    # test_loader = DataLoader(input_size=model.input_size, dataset=test_dataset, batch_size=100, shuffle=False)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=reg)
    losses = []
    for epoch in range(epochs):

        # train loop
        losses += train(epoch, train_loader, model, optimizer, criterion)
        # losses
        # validation loop
        # validate(epoch, test_loader, model, criterion)
    if not os.path.exists('./model_weights/'):
        os.makedirs('./model_weights/')

    plt.plot(losses)
    plt.xlabel('Batch Iterations')
    plt.ylabel('Loss')
    plt.show()

    torch.save(model.state_dict(), f'./model_weights/{model_name}{datetime.now().strftime("%m-%d-%y-%H-%M-%S")}.pth')
