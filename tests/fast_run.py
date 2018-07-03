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

TRAINING_DATA_DIR = './data/preprocessed/MIREX/*.png'

def test():
    data_provider = ImageDataProvider(TRAINING_DATA_DIR, data_suffix = '_in.png', mask_suffix = '_out.png')
    print(data_provider.channels)
    net = Unet(
        num_channels = data_provider.channels, 
        num_classes = data_provider.n_class, 
        layers = 3, 
        features_root = 64,
        cost_kwargs = dict(regularizer = 0.001),
    )
    trainer = Trainer(net, optimizer = 'momentum', opt_kwargs = dict(momentum = 0.2))
    path = trainer.train(
        data_provider,
        './unet_trained',
        training_iters = 32,
        epochs = 1,
        dropout = 0.5,
        display_step = 2
    )

def init():
    tf.reset_default_graph()

    shutil.rmtree('./unet_trained', ignore_errors = True)
    shutil.rmtree('./prediction', ignore_errors = True)

def main():
    init()
    test()

if __name__ == '__main__':
    main()
