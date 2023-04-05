from criteria.Criterion import Criterion
from .MainModel import PretrainedModel, CustomModel
from .modules import Beit, CustomCNN, Resnet, ViT
from loader.Dataloader import ImageDataLoader
import torch
import os
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


def train_validate(epoch, data_loader, model, optimizer, criterion):

    model.train()
    train_losses = AverageMeter()
    val_losses = AverageMeter()

    train_l = []
    val_l = []
    split = int(0.75 * len(data_loader))

    for idx, (data, target) in enumerate(data_loader):

        data = data.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        if idx > split:
            with torch.no_grad():
                out = model(data)
                loss = criterion(out, target)
                val_l.append(float(loss))
                val_losses.update(loss, out.shape[0])
        else:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_l.append(float(loss))
            train_losses.update(loss, out.shape[0])

        if idx % 1 == 0:
            if idx > split:
                print(('(Validation)\t'
                       'Epoch: [{0}][{1}/{2}]\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                      .format(epoch, idx, len(data_loader), loss=val_losses))
            else:
                print(('(Training)\t'
                       'Epoch: [{0}][{1}/{2}]\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                      .format(epoch, idx, len(data_loader), loss=train_losses))

    return train_l, val_l

def main(model_name='Custom', lr=0.001, reg=0.0001, epochs=2, inputColorSpace='RGB', criterionColorSpace='RGB',
         loss='RMSE'):
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

    data_loader = ImageDataLoader(model.input_size, space=inputColorSpace, batch_size=8)

    criterion = Criterion(loss=loss, space=criterionColorSpace)

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=reg)
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # train loop
        new_train_losses, new_val_losses = train_validate(epoch, data_loader, model, optimizer, criterion)
        train_losses += new_train_losses
        test_losses += new_val_losses
        # validation loop
    if not os.path.exists('./model_weights/'):
        os.makedirs('./model_weights/')

    torch.save(model.state_dict(), f'./model_weights/{model_name}{datetime.now().strftime("%m-%d-%y-%H-%M-%S")}.pth')
    return train_losses, test_losses, train_losses, test_losses
