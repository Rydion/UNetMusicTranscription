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

EPOCHS = 20
TRAIN = False

np.random.seed(98765)

def train_network(net, generator):
    trainer = unet.Trainer(
        net,
        optimizer = 'momentum',
        opt_kwargs = dict(momentum = 0.2)
    )
    trainer.train(
        generator,
        './unet_trained_toy',
        prediction_path = './prediction_toy',
        #training_iters = 32,
        epochs = EPOCHS,
        display_step = 2
    )
    
def predict(net, generator):
    x, y = generator(1)
    prediction = net.predict('./unet_trained_toy/model.ckpt', x)
    mask = prediction#[0, ..., 1] > 0.9
    mask = np.select([prediction <= 0.5, prediction > 0.5], [np.zeros_like(prediction), np.ones_like(prediction)])

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

    fig, ax = plt.subplots(1, 4, figsize = (12, 4))
    ax[0].imshow(x[0, ..., 0], aspect = 'auto', cmap = plt.cm.gray)
    ax[1].imshow(y[0, ..., 1], aspect = 'auto', cmap = plt.cm.gray)
    ax[2].imshow(prediction[0, ..., 1], aspect = 'auto', cmap = plt.cm.gray)
    ax[3].imshow(mask[0, ..., 1], aspect = 'auto', cmap = plt.cm.gray)
    ax[0].set_title('Input')
    ax[1].set_title('Ground truth')
    ax[2].set_title('Prediction')
    ax[3].set_title('Mask')
    fig.tight_layout()
    plt.draw()
    plt.show()

def test(train = True):
    nx = 572
    ny = 572
    generator = image_gen.GrayScaleDataProvider(nx, ny, cnt = 20)
    net = unet.Unet(
        channels = generator.channels,
        n_class = generator.n_class,
        layers = 3,
        features_root = 16
    )

    if train:
        train_network(net, generator)
    predict(net, generator)

def init(remove_previous_net = True):
    if remove_previous_net:
        shutil.rmtree('./unet_trained_toy', ignore_errors = True)
        shutil.rmtree('./prediction_toy', ignore_errors = True)

def main():
    init(remove_previous_net = TRAIN)
    test(TRAIN)

if __name__ == '__main__':
    main()
