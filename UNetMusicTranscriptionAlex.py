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

from unet.Unet import UNetModel

TRANSFORMATION = 'cqt' # stft cqt
DATASET = 'MIREX' # Piano MIREX
TRAINING_DATASET = DATASET + '.' + TRANSFORMATION
DATA_DIR = './data/preprocessed/'
TRAINING_DATA_DIR = os.path.join(DATA_DIR, TRAINING_DATASET)

RESULTS_DIR = './results/'
MODEL_NAME = 'tanh'
DST_DIR = os.path.join(RESULTS_DIR, MODEL_NAME)
MODEL_DST_DIR = os.path.join(DST_DIR, 'unet')
PLOT_DST_DIR = os.path.join(DST_DIR, 'prediction')

IMAGE_FORMAT = '.png'
DATA_SUFFIX = '.in'
MASK_SUFFIX = '.out'

LOAD = True
NUM_ITERATIONS = 500
BATCH_SIZE = 10

def train(sess, dataset, model_dst_dir, plot_dest_dir, load = False):
    iterator = dataset.make_one_shot_iterator()

    input, output = iterator.get_next()
    is_training = tf.placeholder(shape = (), dtype = bool)
    model = UNetModel(
        input,
        output,
        is_training
    )

    if load:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_dst_dir, 'model.ckpt'))
    else:
        sess.run(tf.global_variables_initializer())

    for i in range(NUM_ITERATIONS):
        _, cost = sess.run(
            [model.train_op, model.cost],
            { model.is_training: True }
        )

        if i%10 == 0:
            print('{0}, {1}'.format(i, cost))

        if i%100 == 0:
            save_model(sess, model_dst_dir, global_step = i)
            test_model(sess, model, save_fig = True, id = i, dst_dir = plot_dest_dir)

    save_model(sess, model_dst_dir)
    test_model(sess, model, save_fig = False, id = 'final', dst_dir = plot_dest_dir)

def save_model(sess, dst_dir, global_step = None):
    saver = tf.train.Saver()
    save_path = saver.save(sess, os.path.join(dst_dir, 'model.ckpt'), global_step = global_step)
    print('Model saved to: {0}'.format(save_path))

def test_model(sess, model, save_fig = True, id = None, dst_dir = None):
    x, y = sess.run([model.input, model.output])
    prediction = sess.run([model.prediction], { model.is_training: False, model.input: x })
    plot(x, y, prediction, save = save_fig, id = id, dst_dir = dst_dir)

def get_dataset(src_dir, format, input_suffix, output_suffix, bach_size = 1):
    def parse_files(input_file, output_file):
        input_img_string = tf.read_file(input_file)
        output_img_string = tf.read_file(output_file)

        input_img = tf.image.decode_png(input_img_string, channels = 1, dtype = tf.uint8)
        output_img = tf.image.decode_png(output_img_string, channels = 1, dtype = tf.uint8)

        input_img = tf.cast(input_img, tf.float32)/255
        output_img = tf.cast(output_img, tf.float32)/255

        return input_img, output_img

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
    return dataset.batch(bach_size).repeat()

def init():
    if not os.path.isdir(DST_DIR):
        os.makedirs(MODEL_DST_DIR)
        os.makedirs(PREDICTION_DST_DIR)

    tf.reset_default_graph()
    return tf.Session()

def plot(x, y, prediction, save = False, id = 0, dst_dir = None):
    print([np.amin(prediction), np.amax(prediction)])

    x = x[0, ..., 0]
    y = y[0, ..., 0]
    prediction = prediction[0][0, ..., 0]

    fig, ax = plt.subplots(1, 3, figsize = (4*3, 16), dpi = 32)
    ax[0].imshow(x, vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
    ax[1].imshow(y, vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
    ax[2].imshow(prediction, aspect = 'auto', cmap = plt.cm.gray)

    if save:
        fig.savefig(os.path.join(dst_dir, 'test.{0}.png'.format(id)))
        plt.close(fig)
    else:
        plt.draw()
        plt.show()

def main():
    sess = init()
    dataset = get_dataset(
        TRAINING_DATA_DIR,
        IMAGE_FORMAT,
        DATA_SUFFIX,
        MASK_SUFFIX,
        bach_size = BATCH_SIZE
    )
    train(
        sess,
        dataset,
        MODEL_DST_DIR,
        PLOT_DST_DIR,
        load = LOAD
    )

if __name__ == '__main__':
    main()
