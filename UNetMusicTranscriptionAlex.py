'''
created: 2018-05-31
author: Adrian Hintze @Rydion
'''

import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from unet_alex.Unet import UNetModel

DATASET = 'MIREX' # Piano MIREX
TRANSFORMATION = 'cqt' # stft cqt
TRAINING_DATASET = DATASET + '.' + TRANSFORMATION
TRAINING_DATA_DIR = os.path.join('./data/preprocessed/', TRAINING_DATASET)
DST_DIR = './results/'
IMAGE_FORMAT = '.png'
DATA_SUFFIX = '.in'
MASK_SUFFIX = '.out'
NUM_EPOCHS = 100
TRAIN = True
NUM_TESTS = 5

def get_dataset(src_dir, input_suffix, output_suffix, bach_size = 32):
    def parse_files(input_file, output_file):
        input_img = tf.image.decode_image(tf.read_file(input_file))
        output_img = tf.image.decode_image(tf.read_file(output_file))
        return input_img, output_img

    input_files = []
    output_files = []
    for file in os.listdir(src_dir):
         file_name, file_extension = os.path.splitext(file)
         if file_name.endswith(input_suffix):
             input_file_name = file_name
             input_file = file
             output_file_name = os.path.splitext(file_name)[0] + output_suffix
             output_file = output_file_name + file_extension
             if os.path.isfile(os.path.join(src_dir, output_file)):
                 input_files.append(os.path.join(src_dir, input_file))
                 output_files.append(os.path.join(src_dir, output_file))

    dataset = tf.data.Dataset.from_tensor_slices((input_files, output_files))
    dataset = dataset.map(parse_files)
    return dataset.shuffle(buffer_size = 1000).batch(bach_size).repeat(10)

def init():
    tf.reset_default_graph()
    return tf.Session()


def main():
    sess = init()
    ds = get_dataset(TRAINING_DATA_DIR, DATA_SUFFIX, MASK_SUFFIX)
    input, output = ds.make_one_shot_iterator().get_next()
    x, y = sess.run([input, output])

    fig, ax = plt.subplots(1, 2, figsize = (4, 16), dpi = 32)
    ax[0].imshow(x[0, ..., 0], aspect = 'auto', cmap = plt.cm.gray)
    ax[1].imshow(y[0, ..., 0], aspect = 'auto', cmap = plt.cm.gray)
    plt.draw()
    plt.show()

if __name__ == '__main__':
    main()
