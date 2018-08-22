'''
created: 2018-05-31
author: Adrian Hintze @Rydion
'''

import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def train(sess, dataset):
    iterator = dataset.make_one_shot_iterator()
    input, output = iterator.get_next()
    #x, y = sess.run([input, output])
    is_training = tf.placeholder(shape = (), dtype = bool)

    model = UNetModel(
        input,
        output,
        is_training
    )

    sess.run(tf.global_variables_initializer())

    for i in range(10):
        _, cost = sess.run(
            [model.train_op, model.cost],
            { model.is_training: True }
        )
        print('            , {0}, {1}'.format(i, cost))

    x, y = sess.run([input, output])
    prediction = sess.run([model.prediction], { is_training: False, model.input: x })

    plot(x, y, prediction)

def get_dataset(src_dir, format, input_suffix, output_suffix, bach_size = 8):
    def parse_files(input_file, output_file):
        input_img_string = tf.read_file(input_file)
        output_img_string = tf.read_file(output_file)

        input_img = tf.image.decode_image(input_img_string)
        output_img = tf.image.decode_image(output_img_string)

        input_img = tf.cast(input_img, tf.float32)
        output_img = tf.cast(output_img, tf.float32)
        return tf.reshape(input_img, [1, 512, 128, 1]), tf.reshape(output_img, [1, 512, 128, 1])

    input_files = []
    output_files = []
    for file in os.listdir(src_dir):
        file_name, file_extension = os.path.splitext(file)
        if file_extension != format:
            continue
        if not file_name.endswith(input_suffix):
            continue

        input_file_name = file_name
        input_file = file
        output_file_name = os.path.splitext(file_name)[0] + output_suffix
        output_file = output_file_name + file_extension
        if os.path.isfile(os.path.join(src_dir, output_file)):
            input_files.append(os.path.join(src_dir, input_file))
            output_files.append(os.path.join(src_dir, output_file))

    dataset = tf.data.Dataset.from_tensor_slices((input_files, output_files))
    dataset = dataset.map(parse_files)
    return dataset.batch(1).repeat()

def init():
    tf.reset_default_graph()
    return tf.Session()

def plot(x, y, prediction):
    prediction = prediction[0]
    fig, ax = plt.subplots(1, 3, figsize = (4, 16), dpi = 32)
    ax[0].imshow(x[0, ..., 0], aspect = 'auto', cmap = plt.cm.gray)
    ax[1].imshow(y[0, ..., 0], aspect = 'auto', cmap = plt.cm.gray)
    ax[2].imshow(prediction[0, ..., 0], vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
    plt.draw()
    plt.show()

def main():
    sess = init()
    dataset = get_dataset(TRAINING_DATA_DIR, IMAGE_FORMAT, DATA_SUFFIX, MASK_SUFFIX)
    train(sess, dataset)

if __name__ == '__main__':
    main()
