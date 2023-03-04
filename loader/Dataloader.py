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
    def __init__(self, inputShape, space='LAB', batch_size=32, shuffle=True):
        self.dataset = ImageDataset(inputShape, shuffle=shuffle)
        self.batch_size = batch_size
        self.space = space
        self.i = 0
        # self.inProfile = ImageCms.createProfile(
        #     colorSpace='RGB')
        # self.outProfile = ImageCms.createProfile(
        #     colorSpace=self.space)
        # self.colorTransform = ImageCms.buildTransform(
        #     inputProfile=self.inProfile,
        #     outputProfile=self.outProfile,
        #     inMode='RGB',
        #     outMode=space
        # )
        if space == 'LAB':
            self.colorTransform = cv2.COLOR_RGB2LAB
            self.colorTransformInverse = cv2.COLOR_LAB2RGB
            self.imgOffset = np.array([0, 128, 128])
            self.labToSpace = None

    def __iter__(self):
        return self

    def transform(self, image):

        # imageSpace = ImageCms.applyTransform(
        #     im=image,
        #     transform=self.colorTransform
        # )

        # imageSpace = cv2.cvtColor(image, self.colorTransform)

        # imagePositive = imageSpace + self.imgOffset



        X = image
        # imageLab = color.rgb_to_lab(image)

        # temp = np.random.random() - 0.5
        # tint = np.random.random() - 0.5

        # offWhite = np.array([1, ])

        y = torch.tensor([0,0,0])

        return X, y

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
        return torch.from_numpy(np.stack(X).permute(0,3,1,2)), torch.stack(y)


if __name__ == '__main__':
    loader = ImageDataLoader((128, 128))
    for batch in loader:
        print(batch)
    # print(len(loader))
    # print(loader[0])
    # print(loader[-1])