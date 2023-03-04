# from torch.utils.data import DataLoader
from loader.Dataset import ImageDataset
import torch
# from torchvision import transforms
import numpy as np
# from kornia import color
# from PIL import ImageCms
# from skimage.color import rgb2lab, lab2rgb
import cv2
import matplotlib.pyplot as plt

def convert(image, conversions):
    if conversions is None:
        return image
    if type(conversions) is not list:
        return cv2.cvtColor(image, conversions)
    for conversion in conversions:
        image = cv2.cvtColor(image, conversion)
    return image

class ImageDataLoader:
    def __init__(self, inputShape, space='RGB', batch_size=32, shuffle=True):
        self.dataset = ImageDataset(inputShape, shuffle=shuffle)
        self.batch_size = batch_size
        self.space = space
        self.i = 0
        self.channelOffset = 0
        if space == 'RGB':
            self.colorTransform = None
            self.colorTransformInverse = None
            self.grey = np.array([128,128,128]) / 255
            self.labToSpace = cv2.COLOR_LAB2RGB
        elif space == 'XYZ':
            self.colorTransform = cv2.COLOR_RGB2XYZ
            self.colorTransformInverse = cv2.COLOR_XYZ2RGB
            self.grey = np.array([20.52,21.59,23.51])
            self.labToSpace = [cv2.COLOR_LAB2RGB, cv2.COLOR_RGB2XYZ]

        stdRadius = 5
        radius = 128
        self.std = radius / stdRadius

    def __len__(self):
        return len(self.dataset)
    def __iter__(self):
        return self

    def transform(self, image):

        imageInSpace = convert(image, self.colorTransform)
        imageInSpace = imageInSpace.astype(np.float32)
        h, w, c = imageInSpace.shape

        temp = np.random.normal(0, self.std)
        tint = np.random.normal(0, self.std)
        filt = np.array([50, temp, tint]).astype(np.float32)

        if self.labToSpace is not None:
            filtInSpace = convert(filt.reshape(1,1,c), self.labToSpace).reshape(c)
        else:
            filtInSpace = filt

        imageInSpace += self.channelOffset
        scalar = filtInSpace / self.grey
        scalar /= np.sum(scalar) / c
        imageInSpace *= scalar
        imageInSpace -= self.channelOffset

        rgbOut = convert(imageInSpace, self.colorTransformInverse)

        y = torch.from_numpy(convert(filt.reshape(1,1,c), cv2.COLOR_LAB2RGB).reshape(c))

        return rgbOut, y

    def __next__(self):
        if self.i >= len(self.dataset):
            self.i = 0
            raise StopIteration
        X = []
        y = []
        for j in range(self.batch_size):
            if self.i >= len(self.dataset):
                break
            rawImage = self.dataset[self.i]
            transformed, label = self.transform(rawImage)
            X.append(transformed)
            y.append(label)
            self.i += 1
        return torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2), torch.stack(y)

if __name__ == '__main__':
    loader = ImageDataLoader((128, 128), space='XYZ')
    for X, y in loader:
        for i in range(len(X)):
            plt.imshow(X[i].permute(1,2,0))
            plt.show()
            second = np.ones((50,100,3)).astype(np.float32)
            second[:,:50,:] = 0.5
            second[:,50:,:] *= y[i].numpy()
            plt.imshow(second)
            plt.show()
            # print(batch[1][i])
        print(X, y)
    # print(len(loader))
    # print(loader[0])
    # print(loader[-1])