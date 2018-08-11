'''
created: 2018-05-31
author: Adrian Hintze @Rydion
'''

import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tf_unet.unet import Unet, Trainer
from tf_unet.image_util import ImageDataProvider

IMAGE_FORMAT = '.png'
TRAINING_DATA_DIR = './data/preprocessed/Piano/*'
DATA_SUFFIX = '_in' + IMAGE_FORMAT
MASK_SUFFIX = '_out' + IMAGE_FORMAT
EPOCHS = 25
TRAIN = True
TESTS = 5

def train_network(net, data_provider):
    '''
    x, y = data_provider(1)
    fig, ax = plt.subplots(1, 3, figsize = (12, 4))
    ax[0].imshow(x[0, ..., 0], aspect = 'auto')
    ax[1].imshow(y[0, ..., 1], aspect = 'auto', cmap = plt.cm.gray)
    plt.draw()
    plt.show()
    
    print(np.shape(x))
    print(np.shape(y))
    print(np.amin(x))
    print(np.amax(x))
    print(np.amin(y))
    print(np.amax(y))
    print(data_provider.channels)
    print(data_provider.n_class)
    '''
    #return
    trainer = Trainer(
        net,
        optimizer = 'momentum',
        opt_kwargs = dict(momentum = 0.2)
    )
    trainer.train(
        data_provider,
        './unet_trained_fast',
        prediction_path = './prediction_fast',
        training_iters = 32,
        epochs = EPOCHS,
        display_step = 2
    )

def predict(net, data_provider, tests = 1):
    for i in range(tests):
        x, y = data_provider(1)
        prediction = net.predict('./unet_trained_fast/model.ckpt', x)
        mask = np.select([prediction <= 0.5, prediction > 0.5], [np.zeros_like(prediction), np.ones_like(prediction)])
        #mask = np.zeros(np.shape(prediction), dtype = np.float32)
        #mask[:, :, 0] = prediction[:, :, 0] >= 0.9
        #mask[:, :, 1] = prediction[:, :, 1] >= 0.9
        #mask = prediction > 0.9

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
        plt.draw()
        plt.show()

def test(train = True):
    data_provider = ImageDataProvider(
        TRAINING_DATA_DIR,
        data_suffix = DATA_SUFFIX,
        mask_suffix = MASK_SUFFIX
    )
    net = Unet(
        channels = data_provider.channels,
        n_class = data_provider.n_class,
        layers = 4,
        features_root = 16
    )

    if train:
        train_network(net, data_provider)
    predict(net, data_provider, tests = TESTS)

def init(remove_previous_net = True):
    if remove_previous_net:
        shutil.rmtree('./unet_trained_fast', ignore_errors = True)
        shutil.rmtree('./prediction', ignore_errors = True)

def main():
    init(remove_previous_net = TRAIN)
    test(TRAIN)

if __name__ == '__main__':
    main()
