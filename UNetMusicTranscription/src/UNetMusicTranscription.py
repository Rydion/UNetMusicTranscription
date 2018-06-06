'''
created on 2018-05-31
author: Adrian Hintze @Rydion
'''

from __future__ import division, print_function

import tensorflow as tf
from unet.Unet import Unet
from unet.Trainer import Trainer

from tf_unet import image_gen
from tf_unet import util
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.rcParams['image.cmap'] = 'gist_earth'


def test():
    nx = 572
    ny = 572

    generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)

    x_test, y_test = generator(1)

    fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
    ax[0].imshow(x_test[0,...,0], aspect="auto")
    ax[1].imshow(y_test[0,...,1], aspect="auto")

    net = Unet(numChannels = generator.channels, numClasses = generator.n_class, layers = 3, featuresRoot = 16)

    trainer = Trainer(net, optimizer = 'momentum', optKwargs = dict(momentum = 0.2))
    path = trainer.train(generator, "./unet_trained", training_iters = 20, epochs = 10, display_step = 2)

def init():
    tf.reset_default_graph()

def main():
    init()
    test()

if __name__ == '__main__':
    main()
