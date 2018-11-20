'''
author: Adrian Hintze
'''

import os
import argparse
import time
import shutil
import pickle
import warnings
import configparser
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from madmom.evaluation.notes import NoteEvaluation, NoteMeanEvaluation
from unet.Unet import UNetModel
from utils.Preprocessor import Preprocessor
from utils.functions import collapse_array

# Remove unnecessary tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Suppress unnecessary warnings from libraries
warnings.filterwarnings('ignore')

class Wrapper(object):
    def __init__(
        self,
        sess,
        dataset_src_dir,
        model_src_dir,
        img_format,
        input_suffix,
        output_suffix,
        gt_suffix,
        samples_per_second,
        batch_size,
        num_epochs,
        weight,
        kernel_size,
        state = None
    ):
        self.sess = sess
        self._img_format = img_format
        self._num_epochs = num_epochs
        self._samples_per_second = samples_per_second

        self.training_dataset_size, self.validation_dataset_size, self.test_dataset_size, \
        self.training_dataset, self.validation_dataset, self.test_dataset = self._get_datasets(
            dataset_src_dir,
            self._img_format,
            input_suffix,
            output_suffix,
            gt_suffix,
            batch_size = batch_size
        )
   
        self.handle = tf.placeholder(tf.string, shape = [])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.training_dataset.output_types, self.training_dataset.output_shapes)
        self.training_handle = sess.run(self.training_dataset.make_one_shot_iterator().string_handle())
        self.validation_handle = sess.run(self.validation_dataset.make_one_shot_iterator().string_handle())
        self.test_handle = sess.run(self.test_dataset.make_one_shot_iterator().string_handle())

        self.input, self.output, self.ground_truth, self.file_name = self.iterator.get_next()
        self.is_training = tf.placeholder(dtype = bool, shape = ())
        self.model = UNetModel(
            self.input,
            self.output,
            self.is_training,
            weight,
            kernel_size
        )

        if state == None:
            self.state = {
                'epoch': 0,
                'cost': [(0, 0)] # Add an initial element to make the array start at 1 as we will start at epoch 1
            }
            sess.run(tf.global_variables_initializer())
        else:
            self.state = state
            self._load_model(model_src_dir, global_step = self.state['epoch'])

    def train(self, dst_dir, model_dst_dir, training_plot_dst_dir, validation_plot_dst_dir, early_stop_epochs = 20):
        i = 0
        samples = 0
        epoch_cost = 0
        lowest_validation_cost = np.inf
        early_stop_counter = 0

        while True:
            if self.state['epoch'] >= self._num_epochs:
                break

            x, y, prediction, cost, _ = self.sess.run(
                [self.input, self.output, self.model.prediction, self.model.cost, self.model.train_op],
                feed_dict = { self.is_training: True, self.handle: self.training_handle }
            )

            i = i + 1
            samples = samples + np.shape(x)[0]
            epoch_cost = epoch_cost + cost

            # each epoch
            if samples == self.training_dataset_size:
                self.state['epoch'] = self.state['epoch'] + 1

                self._save_model(model_dst_dir, global_step = self.state['epoch'])
                self._plot(x, y, prediction, save = True, id = 'epoch-{0}'.format(self.state['epoch']), dst_dir = training_plot_dst_dir)

                training_cost = epoch_cost/i

                epoch_test_plot_dst_dir = os.path.join(validation_plot_dst_dir, str(self.state['epoch']))
                plot = self.state['epoch']%10 == 0
                if plot and not os.path.exists(epoch_test_plot_dst_dir):
                    os.makedirs(epoch_test_plot_dst_dir)
                validation_cost = self.validate(plot = plot, plot_dest_dir = epoch_test_plot_dst_dir)

                # Update results and save
                if len(self.state['cost']) == self.state['epoch']:
                    self.state['cost'].append((training_cost, validation_cost))
                else:
                    self.state['cost'][self.state['epoch']] = (training_cost, validation_cost)
                with open(os.path.join(dst_dir, 'results.pkl'), 'wb') as f:
                    pickle.dump(self.state, f, pickle.HIGHEST_PROTOCOL)

                print('Epoch {0} finished. Training error: {1}. Validation error: {2}'.format(self.state['epoch'], training_cost, validation_cost))

                change_threshold = 0.001
                if np.abs(lowest_validation_cost - validation_cost) > change_threshold:
                    lowest_validation_cost = validation_cost
                    early_stop_counter = 0
                else:
                    early_stop_counter = early_stop_counter + 1

                if early_stop_counter == early_stop_epochs:
                    print('Early stop at epoch {0}.'.format(self.state['epoch']))
                    break

                i = 0
                samples = 0
                epoch_cost = 0

        self._save_model(model_dst_dir)

    def validate(self, plot = False, plot_dest_dir = None):
        cost, _ = self._test(self.validation_handle, plot = plot, plot_dest_dir = plot_dest_dir)
        return cost

    def test(self, plot = False, save = False, plot_dest_dir = None):
        return self._test(self.test_handle, plot = plot, save = save, plot_dest_dir = plot_dest_dir, evaluate = True)

    def _test(self, handle, plot = False, save = False, plot_dest_dir = None, evaluate = False):
        i = 0
        samples = 0
        total_cost = 0
        evaluations = {
            '0.1': [],
            '0.07': [],
            '0.05': [],
            '0.03': [],
            '0.01': [],
        }
        while True:
            x, y, gt, file_name = self.sess.run(
                [self.input, self.output, self.ground_truth, self.file_name],
                feed_dict = { self.handle: handle }
            )
            file_name = file_name[0].decode()
            prediction, cost = self.sess.run(
                [self.model.prediction, self.model.cost],
                feed_dict = { self.is_training: False, self.input: x, self.handle: handle }
            )

            i = i + 1
            samples = samples + np.shape(x)[0]
            total_cost = total_cost + cost

            if plot:
                self._plot(x, y, prediction, save = True, id = file_name, dst_dir = plot_dest_dir)
            if save or evaluate:
                gt = gt[0, ..., 0]
                prediction = prediction[0, ..., 0]
                prediction = self._threshold_probability(prediction)
                predicted_gt_float = self._onset_detection(prediction)
                if save:
                    img = Image.fromarray((predicted_gt_float*255).astype(np.uint8), 'L')
                    dst_file = os.path.join(plot_dest_dir, '{0}{1}{2}'.format(file_name, '.p', self._img_format))
                    img.save(dst_file)
                    img = Image.fromarray((gt*255).astype(np.uint8), 'L')
                    dst_file = os.path.join(plot_dest_dir, '{0}{1}{2}'.format(file_name, '.gt', self._img_format))
                    img.save(dst_file)
                if evaluate:
                    windows = [0.1, 0.07, 0.05, 0.03, 0.01]
                    for w in windows:
                        evaluations[str(w)].append(self._evaluate_note_onset(
                            (predicted_gt_float).astype(np.uint8),
                            (gt).astype(np.uint8),
                            1/self._samples_per_second,
                            window = w
                        ))

            # one epoch
            if samples == self.test_dataset_size:
                break

        for key in evaluations:
            eval = NoteMeanEvaluation(evaluations[key])
            eval = {
                'num_annotations': eval.num_annotations,
                'num_tp': eval.num_tp,
                'num_fp': eval.num_fp,
                'num_fn': eval.num_fn,
                'precision': eval.precision,
                'recall': eval.recall,
                'fmeasure': eval.fmeasure,
                'accuracy': eval.accuracy,
                'mean_error': eval.mean_error*1000.0,
                'std_error': eval.std_error*1000.0
            }
            evaluations[key] = eval

        return total_cost/i, evaluations

    def _evaluate_note_onset(self, predicted_gt, gt, sample_length, window):
        def format(pianoroll, sample_length):
            result = []
            for i in range(np.shape(pianoroll)[0]):
                note_on = False
                for j in range(np.shape(pianoroll)[1]):
                    if pianoroll[i, j] == 1:
                        if note_on:
                            result[-1][2] = result[-1][2] + sample_length
                            continue

                        note_on = True
                        result.append([sample_length*j, i, sample_length, 127])
                    else:
                        note_on = False
            return result

        predicted_gt_onsets = format(predicted_gt, sample_length)
        gt_onsets = format(gt, sample_length)

        return NoteEvaluation(predicted_gt_onsets, gt_onsets, window = window)

    def _threshold_probability(self, x, p = 0.8):
        return (x > p).astype(x.dtype)

    def _onset_detection(self, x):
        return collapse_array(x, (96, np.shape(x)[1]), np.shape(x)[0]//96, 0)

    def _get_datasets(self, src_dir, format, input_suffix, output_suffix, gt_suffix, batch_size = 1):
        def get_dataset(src_dir, format, input_suffix, output_suffix, gt_suffix, bach_size):
            def parse_files(input_file, output_file, gt_file, file_name):
                def parse_file(file, channels):
                    img_string = tf.read_file(file)
                    img = tf.image.decode_png(img_string, channels = channels, dtype = tf.uint8)
                    img = tf.cast(img, tf.float32)/255
                    return img

                channels = 1
                return parse_file(input_file, channels), parse_file(output_file, channels), parse_file(gt_file, channels), file_name

            def get_input_output_files(src_dir, format, input_suffix, output_suffix, gt_suffix):
                input_files = []
                output_files = []
                gt_files = []
                file_names = []
                for file in os.listdir(src_dir):
                    file_name, file_extension = os.path.splitext(file)
                    if file_extension != format:
                        continue
                    if not file_name.endswith(input_suffix):
                        continue

                    file_name_without_suffix = os.path.splitext(file_name)[0]
                    input_file = file
                    output_file = file_name_without_suffix + output_suffix + file_extension
                    gt_file = file_name_without_suffix + gt_suffix + file_extension
                    input_file_path = os.path.join(src_dir, input_file)
                    output_file_path = os.path.join(src_dir, output_file)
                    gt_file_path = os.path.join(src_dir, gt_file)
                    if os.path.isfile(output_file_path) and os.path.isfile(gt_file_path):
                        input_files.append(input_file_path)
                        output_files.append(output_file_path)
                        gt_files.append(gt_file_path)
                        file_names.append(file_name_without_suffix)
                return input_files, output_files, gt_files, file_names

            input_files, output_files, gt_files, file_names = get_input_output_files(src_dir, format, input_suffix, output_suffix, gt_suffix)
            dataset = tf.data.Dataset \
                      .from_tensor_slices((input_files, output_files, gt_files, file_names)) \
                      .map(parse_files) \
                      .batch(bach_size) \
                      .shuffle(len(input_files)) \
                      .repeat()

            return len(input_files), dataset

        training_dataset_size, training_dataset = get_dataset(
            os.path.join(src_dir, 'training'),
            format,
            input_suffix,
            output_suffix,
            gt_suffix,
            batch_size
        )
        validation_dataset_size, validation_dataset = get_dataset(
            os.path.join(src_dir, 'validation'),
            format,
            input_suffix,
            output_suffix,
            gt_suffix,
            1
        )
        test_dataset_size, test_dataset = get_dataset(
            os.path.join(src_dir, 'test'),
            format,
            input_suffix,
            output_suffix,
            gt_suffix,
            1
        )
        return training_dataset_size, validation_dataset_size, test_dataset_size, training_dataset, validation_dataset, test_dataset

    def _save_model(self, dst_dir, global_step = None):
        saver = tf.train.Saver()
        return saver.save(self.sess, os.path.join(dst_dir, 'model.ckpt'), global_step = global_step)

    def _load_model(self, src_dir, global_step = None):
        saver = tf.train.Saver()
        model = 'model.ckpt' if global_step == None else 'model.ckpt-{0}'.format(global_step)
        saver.restore(self.sess, os.path.join(src_dir, model))

    def _plot(self, x, y, prediction, save = False, id = 0, dst_dir = None):
        x = x[0, ..., 0]
        y = y[0, ..., 0]
        prediction = prediction[0, ..., 0]

        fig, ax = plt.subplots(1, 10)
        ax[0].imshow(x, vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
        ax[1].imshow(y, vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
        ax[2].imshow(prediction, vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
        ax[3].imshow(prediction > 0.3, vmin = False, vmax = True, aspect = 'auto', cmap = plt.cm.gray)
        ax[4].imshow(prediction > 0.4, vmin = False, vmax = True, aspect = 'auto', cmap = plt.cm.gray)
        ax[5].imshow(prediction > 0.5, vmin = False, vmax = True, aspect = 'auto', cmap = plt.cm.gray)
        ax[6].imshow(prediction > 0.6, vmin = False, vmax = True, aspect = 'auto', cmap = plt.cm.gray)
        ax[7].imshow(prediction > 0.7, vmin = False, vmax = True, aspect = 'auto', cmap = plt.cm.gray)
        ax[8].imshow(prediction > 0.8, vmin = False, vmax = True, aspect = 'auto', cmap = plt.cm.gray)
        ax[9].imshow(prediction > 0.9, vmin = False, vmax = True, aspect = 'auto', cmap = plt.cm.gray)

        if save:
            fig.savefig(os.path.join(dst_dir, '{0}.png'.format(id)))
            plt.close(fig)
        else:
            plt.draw()
            plt.show()

def init(
    data_src_dir,
    dataset_src_dir,
    dst_dir,
    model_dst_dir,
    training_plot_dst_dir,
    test_plot_dst_dir,
    img_format,
    input_suffix,
    output_suffix,
    gt_suffix,
    transformation,
    downsample_rate,
    samples_per_second,
    multiplier,
    load
):
    if not load:
        shutil.rmtree(dst_dir, ignore_errors = True)
        # Without the delay sometimes weird stuff happens when deleting/creating the folder
        time.sleep(1)

        # Crate necessary folders
        os.makedirs(model_dst_dir)
        os.makedirs(training_plot_dst_dir)
        os.makedirs(test_plot_dst_dir)

        # Preprocess data if necessary
        if not os.path.isdir(dataset_src_dir):
            preprocessor = Preprocessor(
                data_src_dir,
                dataset_src_dir,
                img_format,
                input_suffix,
                output_suffix,
                gt_suffix,
                downsample_rate,
                samples_per_second
            )
            preprocessor.preprocess(
                transformation = transformation,
                duration_multiplier = multiplier
            )

    tf.reset_default_graph()
    return tf.Session()

def main(
    data_src_dir,
    dataset_src_dir,
    dst_dir,
    model_dst_dir,
    training_plot_dst_dir,
    test_plot_dst_dir,
    img_format,
    input_suffix,
    output_suffix,
    gt_suffix,
    transformation,
    downsample_rate,
    samples_per_second,
    multiplier,
    load,
    train,
    batch_size,
    num_epochs,
    weight,
    kernel_size
):
    sess = init(
        data_src_dir,
        dataset_src_dir,
        dst_dir,
        model_dst_dir,
        training_plot_dst_dir,
        test_plot_dst_dir,
        img_format,
        input_suffix,
        output_suffix,
        gt_suffix,
        transformation,
        downsample_rate,
        samples_per_second,
        multiplier,
        load
    )

    state = None
    if load:
        with open(os.path.join(dst_dir, 'results.pkl'), 'rb') as f:
            state = pickle.load(f)
    wrapper = Wrapper(
        sess,
        dataset_src_dir,
        model_dst_dir,
        img_format,
        input_suffix,
        output_suffix,
        gt_suffix,
        samples_per_second,
        batch_size,
        num_epochs,
        weight,
        kernel_size,
        state = state
    )

    if train:
        wrapper.train(dst_dir, model_dst_dir, training_plot_dst_dir, test_plot_dst_dir)
    test_cost, eval = wrapper.test(save = True, plot_dest_dir = test_plot_dst_dir)

    sess.close()

    state = wrapper.state
    training_cost, validation_cost = state['cost'][-1]
    results = {
        'training_cost': training_cost,
        'validation_cost': validation_cost,
        'eval': eval
    }
    with open(os.path.join(dst_dir, 'test.txt'), 'w') as text_file:
        text_file.write(str(results))
    with open(os.path.join(dst_dir, 'test.pkl'), 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    return training_cost, validation_cost, test_cost, eval

def grid_search(
    data_src_dir,
    dataset_src_dir,
    dst_dir,
    img_format,
    input_suffix,
    output_suffix,
    gt_suffix,
    transformation,
    downsample_rate,
    samples_per_second,
    multiplier,
    load,
    train,
    batch_size,
    num_epochs,
    weights = [35],
    kernel_sizes = [(5, 5)]
):
    dst_dir = dst_dir + '.grid-search'

    i = 0
    best_model = {
        'iteration': i,
        'params': {
            'ks': kernel_sizes[0],
            'w': weights[0]
        },
        'training_cost': 0,
        'validation_cost': np.inf,
        'test_cost': 0,
        'fscore': 0,
        'eval': None
    }
    best_eval = {
        'iteration': i,
        'params': {
            'ks': kernel_sizes[0],
            'w': weights[0]
        },
        'training_cost': 0,
        'validation_cost': 0,
        'test_cost': 0,
        'fscore': np.inf,
        'eval': None
    }
    for ks in kernel_sizes:
        for w in weights:
            print('Grid Search Iteration {0}: ks {1}, w {2}'.format(i, ks, w))
            it_dst_dir = os.path.join(dst_dir, 'ks-{0}.w-{1}'.format(ks, w))
            it_training_cost, it_validation_cost, it_test_cost, it_eval = main(
                data_src_dir,
                dataset_src_dir,
                it_dst_dir,
                os.path.join(it_dst_dir, 'unet'),
                os.path.join(it_dst_dir, 'training-prediction'),
                os.path.join(it_dst_dir, 'test-prediction'),
                img_format,
                input_suffix,
                output_suffix,
                gt_suffix,
                transformation,
                downsample_rate,
                samples_per_second,
                multiplier,
                load,
                train,
                batch_size,
                num_epochs,
                w,
                ks
            )

            it_model = {
                'iteration': i,
                'params': {
                    'ks': ks,
                    'w': w
                },
                'training_cost': it_training_cost,
                'validation_cost': it_validation_cost,
                'test_cost': it_test_cost,
                'fscore': it_eval['0.1']['fmeasure'],
                'eval': it_eval
            }

            with open(os.path.join(dst_dir, '{0}.pkl'.format(i)), 'wb') as f:
                pickle.dump(it_model, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(dst_dir, '{0}.txt'.format(i)), 'w') as text_file:
                text_file.write(str(it_model))

            if it_model['validation_cost'] < best_model['validation_cost']:
                best_model = it_model
            if it_model['eval']['0.1']['fmeasure'] > best_eval['fscore']:
                best_eval = it_model

            i = i + 1

    with open(os.path.join(dst_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dst_dir, 'best_model.txt'), 'w') as text_file:
        text_file.write(str(best_model))
    with open(os.path.join(dst_dir, 'best_eval.pkl'), 'wb') as f:
        pickle.dump(best_eval, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dst_dir, 'best_eval.txt'), 'w') as text_file:
        text_file.write(str(best_eval))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = '0')
    parser.add_argument('--dataset', default = 'mirex')
    parser.add_argument('--duration', type = int, default = 4)
    parser.add_argument('--input_suffix', default = '.in')
    parser.add_argument('--output_suffix', default = '.out')
    parser.add_argument('--gt_suffix', default = '.gt')
    parser.add_argument('--downsample_rate', type = int, default = 16384)
    parser.add_argument('--samples_per_second', type = int, default = 128)
    parser.add_argument('--load', type = bool, default = False)
    parser.add_argument('--train', type = bool, default = True)
    parser.add_argument('--epochs', type = int, default = 35)
    parser.add_argument('--batch_size', type = int, default = 1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    DATASET = args.dataset
    DURATION = args.duration
    TRANSFORMATION = 'cqt'
    IMG_FORMAT = '.png'
    INPUT_SUFFIX = args.input_suffix
    OUTPUT_SUFFIX = args.output_suffix
    GT_SUFFIX = args.gt_suffix

    DOWNSAMPLE_RATE = args.downsample_rate
    SAMPLES_PER_SECOND = args.samples_per_second

    LOAD = args.load
    TRAIN = args.train
    if not TRAIN:
        LOAD = True

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    # paths
    DATA_SRC_DIR = os.path.join('./data/raw/', DATASET) 

    FULL_DATASET = '{0}.{1}.dr-{2}.sps-{3}.dm-{4}'.format(DATASET, TRANSFORMATION, DOWNSAMPLE_RATE, SAMPLES_PER_SECOND, DURATION)
    DATASET_SRC_DIR = os.path.join('./data/preprocessed/', FULL_DATASET)

    MODEL_NAME = '{0}.{1}.dr-{2}.sps-{3}.dm-{4}.ne-{5}.bs-{6}'.format(DATASET, TRANSFORMATION, DOWNSAMPLE_RATE, SAMPLES_PER_SECOND, DURATION, EPOCHS, BATCH_SIZE)
    DST_DIR = os.path.join('./results/', MODEL_NAME)
    MODEL_DST_DIR = os.path.join(DST_DIR, 'unet')
    TRAINING_PLOT_DST_DIR = os.path.join(DST_DIR, 'training-prediction')
    TEST_PLOT_DST_DIR = os.path.join(DST_DIR, 'test-prediction')

    grid_search(
        DATA_SRC_DIR,       # Raw
        DATASET_SRC_DIR,    # Processed
        DST_DIR,
        IMG_FORMAT,
        INPUT_SUFFIX,
        OUTPUT_SUFFIX,
        GT_SUFFIX,
        TRANSFORMATION,
        DOWNSAMPLE_RATE,
        SAMPLES_PER_SECOND,
        DURATION,
        LOAD,
        TRAIN,
        BATCH_SIZE,
        EPOCHS,
        weights = [30, 35, 40],
        kernel_sizes = [(5, 5), (5, 7), (7, 5), (7, 7)]
    )
