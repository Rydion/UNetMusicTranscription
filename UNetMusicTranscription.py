'''
created: 2018-05-31
author: Adrian Hintze @Rydion
'''

import os
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tf_unet.unet import Unet, Trainer
from tf_unet.image_util import ImageDataProvider

DATASET = 'Piano' # Piano MIREX
TRAINING_DATA_DIR = os.path.join('./data/preprocessed/', DATASET, '*')
DST_DIR = './result/'
IMAGE_FORMAT = '.png'
DATA_SUFFIX = '_in' + IMAGE_FORMAT
MASK_SUFFIX = '_out' + IMAGE_FORMAT
NUM_EPOCHS = 20
TRAIN = True
NUM_TESTS = 1

def train_network(net_dir, net, data_provider):
    trainer = Trainer(
        net,
        optimizer = 'momentum',
        opt_kwargs = dict(momentum = 0.1)
    )
    trainer.train(
        data_provider,
        os.path.join(net_dir, 'unet/'),
        prediction_path = os.path.join(net_dir, 'prediction/'),
        epochs = NUM_EPOCHS,
        training_iters = 32,
        display_step = 2
    )

def predict(net_dir, net, data_provider, tests = 1):
    for i in range(tests):
        x, y = data_provider(1)
        prediction = net.predict(os.path.join(net_dir, 'unet/model.ckpt'), x)
        mask = np.select([prediction <= 0.5, prediction > 0.5], [np.zeros_like(prediction), np.ones_like(prediction)])

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

def test(net_dir, train = True):
    data_provider = ImageDataProvider(
        TRAINING_DATA_DIR,
        data_suffix = DATA_SUFFIX,
        mask_suffix = MASK_SUFFIX
    )
    net = Unet(
        channels = data_provider.channels,
        n_class = data_provider.n_class,
        layers = 4,
        filter_size = 3,
        pool_size = 2
    )

    if train:
        train_network(net_dir, net, data_provider)
    predict(net_dir, net, data_provider, tests = 1 if train else NUM_TESTS)

def init(net_dir, remove_previous_net = True):
    if remove_previous_net:
        shutil.rmtree(os.path.join(net_dir, 'unet/'), ignore_errors = True)
        shutil.rmtree(os.path.join(net_dir, 'prediction/'), ignore_errors = True)

def main():
    init(DST_DIR, remove_previous_net = TRAIN)
    test(DST_DIR, TRAIN)

if __name__ == '__main__':
    main()
