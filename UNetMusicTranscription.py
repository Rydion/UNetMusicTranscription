'''
author: Adrian Hintze
'''

import os
import shutil
import configparser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from unet.Unet import UNetModel
from utils.Preprocessor import Preprocessor

class Wrapper(object):
    def __init__(
        self,
        sess,
        dataset_src_dir,
        image_format,
        input_suffix,
        output_suffix,
        batch_size,
        num_epochs
    ):
        self.sess = sess

        self.training_dataset_size, self.test_dataset_size, self.training_dataset, self.test_dataset = self._get_datasets(
            dataset_src_dir,
            image_format,
            input_suffix,
            output_suffix,
            batch_size = batch_size,
            num_epochs = num_epochs
        )

        self.handle = tf.placeholder(tf.string, shape = [])
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

        sess.run(tf.global_variables_initializer())

    def train(self, model_dst_dir, plot_dest_dir, color = False):
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
                    self._plot(x, y, prediction, color, save = True, id = 'epoch-{0}'.format(epoch), dst_dir = plot_dest_dir)

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

    def test(self, plot = False, plot_dest_dir = None, color = False):
        i = 0
        total_cost = 0
        try:
            while True:
                x, y = self.sess.run([self.input, self.output], feed_dict = { self.handle: self.test_handle })
                prediction, cost = self.sess.run(
                    [self.model.prediction, self.model.cost],
                    feed_dict = { self.model.is_training: False, self.model.input: x, self.handle: self.test_handle }
                )

                i = i + np.shape(x)[0]
                total_cost = total_cost + cost

                if plot:
                    self._plot(x, y, prediction, color, save = True, id = 'sample-{0}'.format(i), dst_dir = plot_dest_dir)

                # one epoch
                if i == self.test_dataset_size:
                    break

        except tf.errors.OutOfRangeError:
            pass
        finally:
            return total_cost/i

    def _get_datasets(self, src_dir, format, input_suffix, output_suffix, batch_size = 1, num_epochs = None):
        def get_dataset(src_dir, format, input_suffix, bach_size, num_epochs = None):
            def parse_files(input_file, output_file):
                def parse_file(file, channels):
                    img_string = tf.read_file(file)
                    img = tf.image.decode_png(img_string, channels = channels, dtype = tf.uint8)
                    img = tf.cast(img, tf.float32)/255
                    return img

                channels = 1
                return parse_file(input_file, channels), parse_file(output_file, channels)

            def get_input_output_files(src_dir, format, input_suffix):
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
                return input_files, output_files

            input_files, output_files = get_input_output_files(src_dir, format, input_suffix)
            dataset = tf.data.Dataset \
                      .from_tensor_slices((input_files, output_files)) \
                      .map(parse_files) \
                      .batch(bach_size) \
                      .shuffle(20) \
                      .repeat(num_epochs)

            return len(input_files), dataset

        training_dataset_size, training_dataset = get_dataset(os.path.join(src_dir, 'training'), format, input_suffix, batch_size, num_epochs = num_epochs)
        test_dataset_size, test_dataset = get_dataset(os.path.join(src_dir, 'test'), format, input_suffix, 1)
        return training_dataset_size, test_dataset_size, training_dataset, test_dataset

    def _save_model(self, dst_dir, global_step = None):
        saver = tf.train.Saver()
        return saver.save(self.sess, os.path.join(dst_dir, 'model.ckpt'), global_step = global_step)

    def _load_model(self, src_dir):
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(src_dir, 'model.ckpt'))

    def _plot(self, x, y, prediction, color, save = False, id = 0, dst_dir = None):
        x = x[0, ..., 0]
        y = y[0, ..., 0]
        prediction = prediction[0, ..., 0]
        mask07 = prediction > 0.7
        mask08 = prediction > 0.8
        mask09 = prediction > 0.9
        mask1 = prediction > 0.99

        fig, ax = plt.subplots(1, 7)
        ax[0].imshow(x, vmin = 0, vmax = 1, aspect = 'auto', cmap = None if color else plt.cm.gray)
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

def init(
    data_src_dir,
    dataset_src_dir,
    model_dst_dir,
    training_plot_dst_dir,
    test_plot_dst_dir,
    img_format,
    input_suffix,
    output_suffix,
    transformation,
    downsample_rate,
    multiplier,
    color
):
    if not os.path.isdir(model_dst_dir):
        os.makedirs(model_dst_dir)
    if not os.path.isdir(training_plot_dst_dir):
        os.makedirs(training_plot_dst_dir)
    if not os.path.isdir(test_plot_dst_dir):
        os.makedirs(test_plot_dst_dir)

    if not os.path.isdir(dataset_src_dir):
        preprocessor = Preprocessor(
            data_src_dir,
            dataset_src_dir,
            img_format,
            input_suffix,
            output_suffix,
            downsample_rate
        )
        preprocessor.preprocess(
            transformation = transformation,
            duration_multiplier = multiplier,
            color = color
        )

    tf.reset_default_graph()
    return tf.Session()

def main(
    data_src_dir,
    dataset_src_dir,
    model_dst_dir,
    training_plot_dst_dir,
    test_plot_dst_dir,
    img_format,
    transformation,
    downsample_rate,
    multiplier,
    color,
    input_suffix,
    output_suffix,
    batch_size,
    num_epochs
):
    sess = init(
        data_src_dir,
        dataset_src_dir,
        model_dst_dir,
        training_plot_dst_dir,
        test_plot_dst_dir,
        img_format,
        input_suffix,
        output_suffix,
        transformation,
        downsample_rate,
        multiplier,
        color
    )

    wrapper = Wrapper(
        sess,
        dataset_src_dir,
        img_format,
        input_suffix,
        output_suffix,
        batch_size,
        num_epochs
    )
    wrapper.train(model_dst_dir, training_plot_dst_dir, color)
    wrapper.test(plot = True, plot_dest_dir = test_plot_dst_dir, color = color)

if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read('conf.ini')

    # global conf
    global_conf = conf['global']
    COLOR = global_conf.getboolean('color')
    DATASET = global_conf['dataset']
    DURATION_MULTIPLIER = int(global_conf['multiplier'])
    TRANSFORMATION = global_conf['transformation']
    IMG_FORMAT = global_conf['format']
    INPUT_SUFFIX = global_conf['input_suffix']
    OUTPUT_SUFFIX = global_conf['output_suffix']

    # preprocessing conf
    process_conf = conf['processing']
    DOWNSAMPLE_RATE = int(process_conf['downsample'])

    # training conf
    training_conf = conf['training']
    BATCH_SIZE = int(training_conf['batch'])
    NUM_EPOCHS = int(training_conf['epochs'])

    # paths
    DATA_SRC_DIR = os.path.join('./data/raw/', DATASET) 

    FULL_DATASET = '{0}.{1}.{2}.{3}'.format(DATASET, TRANSFORMATION, DOWNSAMPLE_RATE, DURATION_MULTIPLIER)
    DATASET_SRC_DIR = os.path.join('./data/preprocessed/', FULL_DATASET)

    MODEL_NAME = '{0}.{1}.{2}.{3}.{4}.{5}.40'.format(DATASET, TRANSFORMATION, DOWNSAMPLE_RATE, DURATION_MULTIPLIER, NUM_EPOCHS, BATCH_SIZE)
    RESULTS_DST_DIR = os.path.join('./results/', MODEL_NAME)
    MODEL_DST_DIR = os.path.join(RESULTS_DST_DIR, 'unet')
    TRAINING_PLOT_DST_DIR = os.path.join(RESULTS_DST_DIR, 'training-prediction')
    TEST_PLOT_DST_DIR = os.path.join(RESULTS_DST_DIR, 'test-prediction')

    main(
        DATA_SRC_DIR,       # Raw
        DATASET_SRC_DIR,    # Processed
        MODEL_DST_DIR,
        TRAINING_PLOT_DST_DIR,
        TEST_PLOT_DST_DIR,
        IMG_FORMAT,
        TRANSFORMATION,
        DOWNSAMPLE_RATE,
        DURATION_MULTIPLIER,
        COLOR,   
        INPUT_SUFFIX,
        OUTPUT_SUFFIX,
        BATCH_SIZE,
        NUM_EPOCHS
    )
