'''
created on 2018-05-31
author: Adrian Hintze @Rydion
'''

from __future__ import division, print_function

import shutil
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from unet.Unet import Unet
from unet.Trainer import Trainer
from tf_unet import image_gen

plt.rcParams['image.cmap'] = 'gist_earth'


def test():
    nx = 300
    ny = nx

    generator = image_gen.GrayScaleDataProvider(nx, ny, cnt = 20)
    net = Unet(num_channels = generator.channels, num_classes = generator.n_class, layers = 3, features_root = 16)
    trainer = Trainer(net, opt_kwargs = dict(momentum = 0.2))
    path = trainer.train(generator, './unet_trained', epochs = 1)

def init():
    tf.reset_default_graph()

    shutil.rmtree('./unet_trained', ignore_errors = True)
    shutil.rmtree('./prediction', ignore_errors = True)

def main():
    init()
    test()

if __name__ == '__main__':
    main()
