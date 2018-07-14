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

#plt.rcParams['image.cmap'] = 'gist_earth'
np.random.seed(98765)

def train():
    nx = 572
    ny = 572

    generator = image_gen.GrayScaleDataProvider(nx, ny, cnt = 20)
    x_test, y_test = generator(1)
    print(np.shape(x_test))
    print(np.shape(y_test))
    print(np.amin(x_test))
    print(np.amax(x_test))
    print(np.amin(y_test))
    print(np.amax(y_test))
    print(generator.channels)
    print(generator.n_class)

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
    x_test, y_test = generator(1)
    prediction = net.predict('./unet_trained_toy/model.ckpt', x_test)
    
    fig, ax = plt.subplots(1, 3, sharex = True, sharey = True, figsize = (12, 5))
    ax[0].imshow(x_test[0, ..., 0], aspect = 'auto')
    ax[1].imshow(y_test[0, ..., 1], aspect = 'auto')
    mask = prediction > 0.9
    mask = prediction
    ax[2].imshow(mask[0, ..., 1], aspect = 'auto')
    ax[0].set_title("Input")
    ax[1].set_title("Ground truth")
    ax[2].set_title("Prediction")
    fig.tight_layout()
    plt.draw()
    plt.show()

def test():
    net, generator = train()
    predict(net, generator)

def init():
    #tf.reset_default_graph()

    shutil.rmtree('./unet_trained_toy', ignore_errors = True)
    shutil.rmtree('./prediction_toy', ignore_errors = True)

def main():
    init()
    test()

if __name__ == '__main__':
    main()
