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

class ImageDataLoader:
    def __init__(self, inputShape, space='RGB', batch_size=32, shuffle=True):
        self.dataset = ImageDataset(inputShape, shuffle=shuffle)
        self.batch_size = batch_size
        self.space = space
        self.i = 0
        self.channelOffset = 0
        if space == 'LAB':
            self.colorTransform = cv2.COLOR_RGB2LAB
            self.colorTransformInverse = cv2.COLOR_LAB2RGB
            self.channelOffset = np.array([0, 128, 128])
            self.grey = np.array([53.59, 128, 128])
            self.labToSpace = None
        if space == 'RGB':
            self.colorTransform = None
            self.colorTransformInverse = None
            self.grey = np.array([128,128,128]) / 255
            self.labToSpace = cv2.COLOR_LAB2RGB

    def __iter__(self):
        return self

    def transform(self, image):

        imageInSpace = cv2.cvtColor(image, self.colorTransform) if self.colorTransform is not None else image
        imageInSpace = imageInSpace.astype(np.float32)
        h, w, c = imageInSpace.shape

        temp = np.random.random() * 50 - 25
        tint = np.random.random() * 50 - 25
        filt = np.array([53.59, temp, tint]).astype(np.float32)

        if self.labToSpace is not None:
            filtInSpace = cv2.cvtColor(filt.reshape(1,1,c), self.labToSpace).reshape(c)
        else:
            filtInSpace = filt

        imageInSpace += self.channelOffset
        scalar = filtInSpace / self.grey
        scalar /= np.sum(scalar) / c
        imageInSpace *= scalar
        imageInSpace -= self.channelOffset

        rgbOut = cv2.cvtColor(image, self.colorTransformInverse) if self.colorTransformInverse is not None else imageInSpace

        y = torch.from_numpy(cv2.cvtColor(filt.reshape(1,1,3), cv2.COLOR_LAB2RGB).reshape(3))

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
        return torch.from_numpy(np.stack(X)).permute(0,3,1,2), torch.stack(y)


if __name__ == '__main__':
    loader = ImageDataLoader((128, 128))
    for batch in loader:
        print(batch)
    # print(len(loader))
    # print(loader[0])
    # print(loader[-1])