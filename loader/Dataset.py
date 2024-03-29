from torch.utils.data import Dataset
import os
from PIL import Image, ImageFile
import multiprocess as mp
import random
import cv2

def resizeImages(baseDir, newDir, imageShape):
    paths = os.listdir(baseDir)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    def resizeImage(name):
        try:
            img = Image.open(os.path.join(baseDir, name))
            resized = img.resize(imageShape, Image.ANTIALIAS)
            resized.save(os.path.join(newDir, name), 'JPEG', quality=100)
            print(f'{name} resized to {imageShape}')
        except Exception:
            print(f'{name} unable to be resized')

    with mp.Pool(None) as p:
        p.map(resizeImage, paths)


class ImageDataset(Dataset):
    def __init__(self, inputShape, shuffle=False, maxLen=None):
        self.inputShape = inputShape
        self.maxLen = maxLen
        h, w = inputShape
        baseDir = os.path.join(os.path.dirname(__file__), f'../data/images')
        self.imageDir = os.path.join(os.path.dirname(__file__), f'../data/images_{h}x{w}')
        if not os.path.exists(self.imageDir):
            print('folder for size', inputShape, 'not found, creating...')
            os.makedirs(self.imageDir)
            resizeImages(baseDir, self.imageDir, inputShape)
        self.names = os.listdir(self.imageDir)
        self.shouldShuffle = shuffle
        self.shuffle()

    def shuffle(self):
        if self.shouldShuffle:
            random.shuffle(self.names)

    def __len__(self):
        if self.maxLen is None:
            return len(self.names)
        return min(len(self.names), self.maxLen)

    def __getitem__(self, i):
        img = cv2.imread(os.path.join(self.imageDir, self.names[i]))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    dataset = ImageDataset((128, 128))
    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])