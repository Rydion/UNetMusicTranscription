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
from utils.Preprocessor import Preprocessor

DATASET = 'MIREX' # Piano MIREX
DURATION_MULTIPLIER = 1 # 1 for 1 second slices, 2 for 0.5 seconds, etc
TRANSFORMATION = 'cqt' # stft cqt
NUM_EPOCHS = 20
BATCH_SIZE = 1

DATA_SRC_DIR = os.path.join('./data/raw/', DATASET) 

DATASET_DIR = './data/preprocessed/'
FULL_DATASET = '{0}.{1}.{2}'.format(DATASET, TRANSFORMATION, DURATION_MULTIPLIER)
FULL_DATASET_DIR = os.path.join(DATASET_DIR, FULL_DATASET)

RESULTS_DIR = './results/'
MODEL_NAME = '{0}-{1}-{2}-{3}-{4}-40'.format(DATASET, TRANSFORMATION, DURATION_MULTIPLIER, NUM_EPOCHS, BATCH_SIZE)
FULL_RESULTS_DIR = os.path.join(RESULTS_DIR, MODEL_NAME)

MODEL_DST_DIR = os.path.join(FULL_RESULTS_DIR, 'unet')
TRAINING_PLOT_DST_DIR = os.path.join(FULL_RESULTS_DIR, 'training-prediction')
TEST_PLOT_DST_DIR = os.path.join(FULL_RESULTS_DIR, 'test-prediction')

IMAGE_FORMAT = '.png'
DATA_SUFFIX = '.in'
MASK_SUFFIX = '.out'

TRAIN = True

class Wrapper(object):
    def __init__(self, sess, load = False, model_src_dir = None):
        self.sess = sess

        self.training_dataset_size, self.test_dataset_size, self.training_dataset, self.test_dataset = self._get_dataset(
            FULL_DATASET_DIR,
            IMAGE_FORMAT,
            DATA_SUFFIX,
            MASK_SUFFIX,
            bach_size = BATCH_SIZE,
            num_epochs = NUM_EPOCHS
        )

        self.handle = tf.placeholder(tf.string, shape = [], name = 'dataset-handle-placeholder')
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.training_dataset.output_types, self.training_dataset.output_shapes)
        self.training_handle = sess.run(self.training_dataset.make_one_shot_iterator().string_handle())
        self.test_handle = sess.run(self.test_dataset.make_one_shot_iterator().string_handle())

        self.input, self.output = self.iterator.get_next()
        self.is_training = tf.placeholder(dtype = bool, shape = ())
        self.model = UNetModel(
            self.input,
            self.output,
            self.is_training
        )

        if load:
            self._load_model(model_src_dir)
        else:
            sess.run(tf.global_variables_initializer())

    def train(self, model_dst_dir, plot_dest_dir):
        i = 0
        epoch = 1
        epoch_cost = 0
        try:
            while True:
                x, y, prediction, cost, _ = self.sess.run(
                    [self.model.input, self.model.output, self.model.prediction, self.model.cost, self.model.train_op],
                    feed_dict = { self.model.is_training: True, self.handle: self.training_handle }
                )

                i = i + np.shape(x)[0]
                epoch_cost = epoch_cost + cost

                # each epoch
                if i == self.training_dataset_size:
                    self._save_model(model_dst_dir, global_step = epoch)
                    self._plot(x, y, prediction, save = True, id = 'epoch-{0}'.format(epoch), dst_dir = plot_dest_dir)

                    training_error = epoch_cost/i
                    test_error = self.test()
                    print('Epoch {0} finished. Training error: {1}. Test error: {2}'.format(epoch, training_error, test_error))

                    i = 0
                    epoch = epoch + 1
                    epoch_cost = 0

        except tf.errors.OutOfRangeError:
            pass
        finally:
            self._save_model(model_dst_dir)

    def test(self, plot = False, plot_dest_dir = None):
        i = 0
        total_cost = 0
        try:
            while True:
                x, y = self.sess.run([self.input, self.output], feed_dict = { self.handle: self.test_handle })
                prediction, cost = self.sess.run(
                    [self.model.prediction, self.model.cost],
                    feed_dict = { self.model.is_training: False, self.model.input: x, self.handle: self.test_handle }
                )

                i = i + 1
                total_cost = total_cost + cost

                if plot:
                    self._plot(x, y, prediction, save = True, id = 'sample-{0}'.format(i), dst_dir = plot_dest_dir)

                # one epoch
                if i == self.test_dataset_size:
                    break

        except tf.errors.OutOfRangeError:
            pass
        finally:
            return total_cost/i

    def _get_dataset(self, src_dir, format, input_suffix, output_suffix, bach_size = 1, num_epochs = None):
        def parse_files(input_file, output_file):
            input_img_string = tf.read_file(input_file)
            output_img_string = tf.read_file(output_file)

            input_img = tf.image.decode_png(input_img_string, channels = 1, dtype = tf.uint8)
            output_img = tf.image.decode_png(output_img_string, channels = 1, dtype = tf.uint8)

            input_img = tf.cast(input_img, tf.float32)/255
            output_img = tf.cast(output_img, tf.float32)/255

            return input_img, output_img

        training_input_files = []
        training_output_files = []
        training_dir = os.path.join(src_dir, 'training')
        for file in os.listdir(training_dir):
            file_name, file_extension = os.path.splitext(file)
            if file_extension != format:
                continue
            if not file_name.endswith(input_suffix):
                continue

            input_file_name = file_name
            input_file = file
            output_file_name = os.path.splitext(file_name)[0] + output_suffix
            output_file = output_file_name + file_extension
            if os.path.isfile(os.path.join(training_dir, output_file)):
                training_input_files.append(os.path.join(training_dir, input_file))
                training_output_files.append(os.path.join(training_dir, output_file))

        test_input_files = []
        test_output_files = []
        test_dir = os.path.join(src_dir, 'test')
        for file in os.listdir(test_dir):
            file_name, file_extension = os.path.splitext(file)
            if file_extension != format:
                continue
            if not file_name.endswith(input_suffix):
                continue

            input_file_name = file_name
            input_file = file
            output_file_name = os.path.splitext(file_name)[0] + output_suffix
            output_file = output_file_name + file_extension
            if os.path.isfile(os.path.join(test_dir, output_file)):
                test_input_files.append(os.path.join(test_dir, input_file))
                test_output_files.append(os.path.join(test_dir, output_file))

        training_dataset_size = len(training_input_files)
        test_dataset_size = len(test_input_files)

        training_dataset = tf.data.Dataset \
                           .from_tensor_slices((training_input_files, training_output_files)) \
                           .map(parse_files)
        test_dataset = tf.data.Dataset \
                       .from_tensor_slices((test_input_files, test_output_files)) \
                       .map(parse_files)

        training_dataset = training_dataset.batch(bach_size).shuffle(20).repeat(num_epochs)
        test_dataset = test_dataset.batch(1).shuffle(20).repeat()

        return training_dataset_size, test_dataset_size, training_dataset, test_dataset

    def _save_model(self, dst_dir, global_step = None):
        saver = tf.train.Saver()
        return saver.save(self.sess, os.path.join(dst_dir, 'model.ckpt'), global_step = global_step)

    def _load_model(self, src_dir):
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(src_dir, 'model.ckpt'))

    def _plot(self, x, y, prediction, save = False, id = 0, dst_dir = None):
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

    if not os.path.isdir(FULL_DATASET_DIR):
        preprocessor = Preprocessor(DATA_SRC_DIR, FULL_DATASET_DIR)
        preprocessor.preprocess(
            transformation = TRANSFORMATION,
            duration_multiplier = DURATION_MULTIPLIER
        )

    tf.reset_default_graph()
    return tf.Session()

def main():
    sess = init(FULL_RESULTS_DIR, MODEL_DST_DIR, TRAINING_PLOT_DST_DIR, TEST_PLOT_DST_DIR)

    if TRAIN:
        wrapper = Wrapper(sess, load = False)
        wrapper.train(MODEL_DST_DIR, TRAINING_PLOT_DST_DIR)
    else:
        wrapper = Wrapper(sess, load = False, model_src_dir = MODEL_DST_DIR)

    wrapper.test(plot = True, plot_dest_dir = TEST_PLOT_DST_DIR)

if __name__ == '__main__':
    main()
