'''
created: 2018-05-31
edited: 2018-07-03
author: Adrian Hintze @Rydion
'''

import shutil
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from unet.Unet import Unet
from unet.Trainer import Trainer
from tf_unet.image_util import ImageDataProvider

plt.rcParams['image.cmap'] = 'gist_earth'

IMAGE_FORMAT = '.bmp'
TRAINING_DATA_DIR = './data/preprocessed/MIREX/*' + IMAGE_FORMAT

def train():
    data_provider = ImageDataProvider(
        TRAINING_DATA_DIR,
        data_suffix = '_in' + IMAGE_FORMAT,
        mask_suffix = '_out' + IMAGE_FORMAT,
        n_class = 3
    )
    print(data_provider.channels)
    net = Unet(
        num_channels = data_provider.channels,
        num_classes = data_provider.n_class,
        layers = 3,
        cost_kwargs = dict(regularizer = 0.001)
    )
    trainer = Trainer(net, optimizer = 'momentum', opt_kwargs = dict(momentum = 0.2))
    path = trainer.train(
        data_provider,
        './unet_trained',
        training_iters = 32,
        epochs = 10,
        dropout = 0.5,
        display_step = 2
    )

    return net, path

def predict(net, path):
    data_provider = ImageDataProvider(TRAINING_DATA_DIR, data_suffix = '_in' + IMAGE_FORMAT, mask_suffix = '_out' + IMAGE_FORMAT, n_class = 3)
    x, y = data_provider(1)
    prediction = net.predict(path, x)

    print('plot')
    fig, ax = plt.subplots(1, 3, figsize = (12, 4))
    ax[0].imshow(x[0, ..., 0], aspect = 'auto')
    ax[1].imshow(y[0, ..., 1], aspect = 'auto')
    ax[2].imshow(prediction[0, ..., 1], aspect = 'auto')
    plt.draw()
    plt.show()

def test():
    net, path = train()
    predict(net, path)

def init():
    tf.reset_default_graph()

    shutil.rmtree('./unet_trained', ignore_errors = True)
    shutil.rmtree('./prediction', ignore_errors = True)

def main():
    init()
    test()

if __name__ == '__main__':
    main()
