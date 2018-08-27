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

DURATION_MULTIPLIER = 2
TRANSFORMATION = 'cqt' # stft cqt
DATASET = 'MIREX' # Piano MIREX
TRAINING_DATASET = '{0}.{1}.{2}'.format(DATASET, TRANSFORMATION, DURATION_MULTIPLIER)
DATA_DIR = './data/preprocessed/'
TRAINING_DATA_DIR = os.path.join(DATA_DIR, TRAINING_DATASET)

NUM_EPOCHS = 10
BATCH_SIZE = 1

RESULTS_DIR = './results/'
MODEL_NAME = '{0}-{1}-{2}-{3}-{4}-30'.format(DATASET, TRANSFORMATION, DURATION_MULTIPLIER, NUM_EPOCHS, BATCH_SIZE)
DST_DIR = os.path.join(RESULTS_DIR, MODEL_NAME)
MODEL_DST_DIR = os.path.join(DST_DIR, 'unet')
TRAINING_PLOT_DST_DIR = os.path.join(DST_DIR, 'training-prediction')
TEST_PLOT_DST_DIR = os.path.join(DST_DIR, 'test-prediction')

IMAGE_FORMAT = '.png'
DATA_SUFFIX = '.in'
MASK_SUFFIX = '.out'

LOAD = False

def train(sess, handle, training_dataset, model_dst_dir, plot_dest_dir, load = False):
    training_iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
    input, output = training_iterator.get_next()
    is_training = tf.placeholder(dtype = bool, shape = ())
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

    i = 0
    training_handle = sess.run(training_dataset.make_one_shot_iterator().string_handle())
    try:
        while True:
            x, y, prediction, cost, _ = sess.run(
                [model.input, model.output, model.prediction, model.cost, model.train_op],
                feed_dict = { model.is_training: True, handle: training_handle }
            )

            if i%10 == 0:
                print('{0}, {1}'.format(i, cost))

            if i%100 == 0:
                save_model(sess, model_dst_dir, global_step = i)
                plot(x, y, prediction, save = True, id = 'train.{0}'.format(i), dst_dir = plot_dest_dir)

            i = i + 1
    except tf.errors.OutOfRangeError:
        pass

    return model

def test(sess, model, handle, test_dataset, plot_dest_dir):
    test_iterator = tf.data.Iterator.from_string_handle(handle, test_dataset.output_types, test_dataset.output_shapes)
    input, output = test_iterator.get_next()

    i = 0
    test_handle = sess.run(test_dataset.make_one_shot_iterator().string_handle())
    try:
        while True:
            x, y = sess.run([input, output], { handle: test_handle })
            prediction, cost = sess.run(
                [model.prediction, model.cost],
                feed_dict = { model.is_training: False, model.input: x, handle: test_handle }
            )

            print('{0}, {1}'.format(i, cost))
            plot(x, y, prediction, save = True, id = 'test.{0}'.format(i), dst_dir = plot_dest_dir)

            i = i + 1
    except tf.errors.OutOfRangeError:
        pass

def save_model(sess, dst_dir, global_step = None):
    saver = tf.train.Saver()
    save_path = saver.save(sess, os.path.join(dst_dir, 'model.ckpt'), global_step = global_step)

def get_dataset(src_dir, format, input_suffix, output_suffix, bach_size = 1, num_epochs = None):
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
    dataset = dataset.shuffle(20)

    dataset_size = len(input_files)
    train_size = int(0.8*dataset_size)
    training_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    training_dataset = training_dataset.batch(bach_size).repeat(num_epochs)
    test_dataset = test_dataset.batch(1)

    return training_dataset, test_dataset

def plot(x, y, prediction, save = False, id = 0, dst_dir = None):
    x = x[0, ..., 0]
    y = y[0, ..., 0]
    prediction = prediction[0, ..., 0]
    mask07 = prediction > 0.7
    mask08 = prediction > 0.8
    mask09 = prediction > 0.9
    mask1 = prediction > 0.99

    fig, ax = plt.subplots(1, 7, figsize = (7*4, 16), dpi = 32)
    ax[0].imshow(x, vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
    ax[1].imshow(y, vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
    ax[2].imshow(prediction, vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
    ax[3].imshow(mask07, aspect = 'auto', cmap = plt.cm.gray)
    ax[4].imshow(mask08, aspect = 'auto', cmap = plt.cm.gray)
    ax[5].imshow(mask09, aspect = 'auto', cmap = plt.cm.gray)
    ax[6].imshow(mask1, aspect = 'auto', cmap = plt.cm.gray)

    if save:
        fig.savefig(os.path.join(dst_dir, '{0}.png'.format(id)))
        plt.close(fig)
    else:
        plt.draw()
        plt.show()

def init(dst_dir, model_dst_dir, training_plot_dst_dir, test_plot_dst_dir):
    if not os.path.isdir(dst_dir):
        os.makedirs(model_dst_dir)
        os.makedirs(training_plot_dst_dir)
        os.makedirs(test_plot_dst_dir)

    tf.reset_default_graph()
    return tf.Session()

def main():
    sess = init(DST_DIR, MODEL_DST_DIR, TRAINING_PLOT_DST_DIR, TEST_PLOT_DST_DIR)

    training_dataset, test_dataset = get_dataset(
        TRAINING_DATA_DIR,
        IMAGE_FORMAT,
        DATA_SUFFIX,
        MASK_SUFFIX,
        bach_size = BATCH_SIZE,
        num_epochs = NUM_EPOCHS
    )

    handle = tf.placeholder(tf.string, shape = [], name = 'dataset-handle-placeholder')
    model = train(
        sess,
        handle,
        training_dataset,
        MODEL_DST_DIR,
        TRAINING_PLOT_DST_DIR,
        load = LOAD
    )

    test(sess, model, handle, test_dataset, TEST_PLOT_DST_DIR)

if __name__ == '__main__':
    main()
