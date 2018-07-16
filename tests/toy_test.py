'''
created: 2018-07-09
edited: 2018-07-09
author: Adrian Hintze @Rydion
'''

import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from utils.functions import binarize

np.random.seed(98765)

def train():
    nx = 572
    ny = 572

    generator = image_gen.GrayScaleDataProvider(nx, ny, cnt = 20)
    net = unet.Unet(
        channels = generator.channels,
        n_class = generator.n_class,
        layers = 3,
        features_root = 16
    )
    trainer = unet.Trainer(
        net,
        optimizer = 'momentum',
        opt_kwargs = dict(momentum = 0.2)
    )
    path = trainer.train(
        generator,
        './unet_trained_toy',
        prediction_path = './prediction_toy',
        training_iters = 32,
        epochs = 1,
        display_step = 2
    )

    return net, generator
    
def predict(net, generator):
    x, y = generator(1)
    prediction = net.predict('./unet_trained_toy/model.ckpt', x)
    mask = prediction#[0, ..., 1] > 0.9

    print(np.shape(x))
    print(np.shape(y))
    print(np.shape(prediction))
    print(np.shape(mask))
    print(np.amin(x))
    print(np.amax(x))
    print(np.amin(y))
    print(np.amax(y))
    print(np.amin(prediction))
    print(np.amax(prediction))
    print(np.amin(mask))
    print(np.amax(mask))

    fig, ax = plt.subplots(1, 4, sharex = True, sharey = True, figsize = (12, 5))
    ax[0].imshow(x[0, ..., 0], aspect = 'auto', cmap = plt.cm.gray)
    ax[1].imshow(y[0, ..., 1], aspect = 'auto', cmap = plt.cm.gray)
    ax[2].imshow(prediction[0, ..., 1], aspect = 'auto', cmap = plt.cm.binary)
    ax[3].imshow(mask[0, ..., 1], aspect = 'auto', cmap = plt.cm.binary)
    ax[0].set_title('Input')
    ax[1].set_title('Ground truth')
    ax[2].set_title('Prediction')
    ax[3].set_title('Mask')
    fig.tight_layout()
    plt.draw()
    plt.show()

def test():
    net, generator = train()
    predict(net, generator)

def init():
    shutil.rmtree('./unet_trained_toy', ignore_errors = True)
    shutil.rmtree('./prediction_toy', ignore_errors = True)

def main():
    init()
    test()

if __name__ == '__main__':
    main()
