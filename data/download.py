import pandas as pd
import os
import requests
from multiprocessing.dummy import Pool


# os.system('wget https://unsplash.com/data/lite/latest -O dataset.zip && mkdir dataset && unzip dataset.zip -d dataset')

df = pd.read_csv('./dataset/photos.tsv000', sep='\t', header=0)

def downloadImage(url):
  try:
    data = requests.get(url).content
  except:
    pass
  id = url.split('/')[-1]
  with open('images/' + id + '.jpg', 'wb') as f:
      f.write(data)
  print(id)

# import os

maxImages = 100

numThreads = 100
pool = Pool(numThreads)

try:
  os.mkdir('images')
except:
  pass

pool.map(downloadImage,df['photo_image_url'][:maxImages])
pool.close()
pool.join()