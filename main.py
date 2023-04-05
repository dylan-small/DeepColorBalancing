from model.train import main
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# if __name__ == "__main__":
#      main()

models = ['ViT', 'Beit', 'Resnet', 'Custom']
inputColorSpaces = ['RGB', 'XYZ']
criterionColorSpaces = ['RGB', 'LAB', 'HSV', 'Hue']
criteria = ['RMSE']

dfName = f'results_{datetime.now()}.csv'
df = None

if __name__ == '__main__':

    for model in models:
        for inputColorSpace in inputColorSpaces:
            for criterionColorSpace in criterionColorSpaces:
                for criterion in criteria:
                    print(f"Running {model} with augmentation color space {inputColorSpace} and criterion color space {criterion} and loss {criterion}")
                    # main should return final loss
                    start = time.time()
                    trainLosses, testLosses, trainAccuracies, testAccuracies = main(model_name = model, lr = 0.0001, reg = 0.0001, epochs = 5, inputColorSpace = inputColorSpace, criterionColorSpace = criterionColorSpace, loss = criterion)
                    duration = time.time() - start


                    x = np.arange(len(trainLosses))
                    plt.plot(x, trainLosses, label = 'train')
                    plt.plot(x, testLosses, label = 'test')
                    plt.title(f"{model} Loss ({inputColorSpace} augmentation and {criterionColorSpace} {criterion} criterion)")
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    lossFilePath = f"./plots/{model}_{inputColorSpace}_{criterionColorSpace}_{criterion}_loss.png"
                    plt.savefig(lossFilePath)

                    plt.plot(x, trainAccuracies, label='train')
                    plt.plot(x, testAccuracies, label='test')
                    plt.title(
                        f"{model} Acc ({inputColorSpace} augmentation and {criterionColorSpace} {criterion} criterion)")
                    plt.xlabel('Epochs')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    accFilePath = f"./plots/{model}_{inputColorSpace}_{criterionColorSpace}_{criterion}_acc.png"
                    plt.savefig(accFilePath)


                    row =  {
                        'Model': model,
                        'Augmentation Color Space': inputColorSpace,
                        'Criterion Color Space': criterionColorSpace,
                        'Criterion': criterion,
                        'Final Train Loss': trainLosses[-1],
                        'Final Test Loss': testLosses[-1],
                        'Final Train Accuracy': trainAccuracies[-1],
                        'Final Test Accuracy': testAccuracies[-1],
                        'Loss Plot': lossFilePath,
                        'Accuracy Plot': accFilePath,
                    }

                    if df is None:
                        df = pd.DataFrame.from_records([row])
                    else:
                        df.append(row)
                    df.to_csv(dfName)


