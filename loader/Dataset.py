from torch.utils.data import Dataset
import os
from PIL import Image
import multiprocess as mp

def resizeImages(baseDir, newDir, imageShape):
    paths = os.listdir(baseDir)

    def resizeImage(name):
        img = Image.open(os.path.join(baseDir, name))
        resized = img.resize(imageShape, Image.ANTIALIAS)
        resized.save(os.path.join(newDir, name), 'JPEG', quality=100)
        print(f'{name} resized to {imageShape}')

    with mp.Pool(None) as p:
        p.map(resizeImage, paths)


class ImageDataset(Dataset):
    def __init__(self, inputShape):
        h, w = inputShape
        baseDir = os.path.join(os.path.dirname(__file__), f'../data/images')
        self.imageDir = os.path.join(os.path.dirname(__file__), f'../data/images_{h}x{w}')
        if not os.path.exists(self.imageDir):
            print('folder for size', inputShape, 'not found, creating...')
            os.makedirs(self.imageDir)
            resizeImages(baseDir, self.imageDir, inputShape)
        self.names = os.listdir(self.imageDir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return Image.open(os.path.join(self.imageDir, self.names[i]))

if __name__ == '__main__':
    dataset = ImageDataset((128, 128))
    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])