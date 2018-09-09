'''
author: Adrian Hintze @Rydion
'''

import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from utils.AudioReader import AudioReader
from utils.Stft import Stft
from utils.CQT import CQT
from utils.Midi import Midi
from utils.functions import grey_scale, binarize, get_chunk_generator

plt.rcParams['figure.figsize'] = [4, 16]
plt.rcParams['figure.dpi'] = 32

# TODO:
#   support other extensions apart from wav
#   make the size of the output/input pair images a parameter

class Preprocessor:
    IMAGE_FORMAT = '.png'
    DOWNSAMPLE_RATE = 8192 #16384 #Hz
    SFFT_WINDOW_LENGTH = 1024
    SFFT_STRIDE = SFFT_WINDOW_LENGTH//2
    FILL_DIGITS = 4
    INPUT_SUFFIX = 'in'
    OUTPUT_SUFFIX = 'out'

    @staticmethod
    def calc_rounded_slice_length(num, dem):
        """
            Calculate the closest that would give a remainder of 0.
            This way we avoid weird results when slicing the spectrogram.
            We will end up with N or N - 1 slices (where N is the number of slices if num%dem == 0).
        """
        remainder = num%dem
        remainder = dem - remainder
        true_len = num + remainder
        return true_len//dem

    def __init__(self, src_dir, dst_dir):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.training_dir = os.path.join(self.dst_dir, 'training')
        self.test_dir = os.path.join(self.dst_dir, 'test')

    def preprocess(self, gen_input = True, gen_output = True, transformation = 'stft', duration_multiplier = 1):
        self._delete_dst_dir()
        self._create_dst_dirs()

        for file in os.listdir(self.src_dir):
            file_name, file_extension = os.path.splitext(file)
            # Search for audio files
            if file_extension != '.wav':
                continue

            # Skip any input file that does not have a corresponding output
            if not os.path.isfile(os.path.join(self.src_dir, file_name + '.mid')):
                continue

            print('Generating input/output for %s.' % file_name)

            # Read original files
            midi = Midi.from_file(os.path.join(self.src_dir, file_name + '.mid'))
            duration = midi.get_length_seconds()

            sample_rate, samples = AudioReader.read_wav(
                os.path.join(self.src_dir, file),
                as_mono = True,
                downsample = True,
                downsample_rate = Preprocessor.DOWNSAMPLE_RATE,
                duration = duration
            )

            if transformation == 'stft':
                spectrogram = Stft.from_audio(
                    sample_rate,
                    samples,
                    window_length = Preprocessor.SFFT_WINDOW_LENGTH,
                    stride = Preprocessor.SFFT_STRIDE
                )
            elif transformation == 'cqt':
                spectrogram = CQT.from_audio(
                    sample_rate,
                    samples
                )
            else:
                raise ValueError('Unknown transformation: ' + transformation + '.')

            subdivisions = int(duration)*duration_multiplier
            spectrogram_img = spectrogram.get_img(subdivisions, 84)
            midi_img = midi.get_img(subdivisions)
            # Same for both images
            slice_length = Preprocessor.calc_rounded_slice_length(np.shape(spectrogram_img)[1], subdivisions)

            if gen_input:
                spectrogram.save(os.path.join(self.dst_dir, file_name + '.spectrogram.png'), subdivisions, 84, color = False)
                chunks = get_chunk_generator(spectrogram_img, slice_length)
                self._save_sliced(chunks, file_name, file_suffix = Preprocessor.INPUT_SUFFIX, binary = False)

            if gen_output:
                midi.save(os.path.join(self.dst_dir, file_name + '.midi.png'), subdivisions)
                chunks = get_chunk_generator(midi_img, slice_length)
                self._save_sliced(chunks, file_name, file_suffix = Preprocessor.OUTPUT_SUFFIX, binary = True)

            # Split into train/test by class
            self._split()

            # Output aesthetics
            print()

    def _delete_dst_dir(self):
        if os.path.exists(self.dst_dir):
            shutil.rmtree(self.dst_dir)
            # Without the delay sometimes weird shit happens when deleting/creating the folder
            time.sleep(1)

    def _create_dst_dirs(self):
        os.makedirs(self.dst_dir)
        os.makedirs(self.training_dir)
        os.makedirs(self.test_dir)

    def _save_sliced(self, chunks, file_name, file_suffix = '', binary = False):
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)

        start = time.clock()
        slice_length = 0
        for i, c in enumerate(chunks):
            # Don't save the last slice if it is smaller than the rest
            if np.shape(c)[1] < slice_length:
                break

            slice_length = np.shape(c)[1]

            if binary:
                c = grey_scale(c)
                c = binarize(c, 200)

            ax.clear()
            ax.axis('off')
            ax.imshow(
                c,
                aspect = 'auto',
                vmin = 0,
                vmax = 255 if binary else 1,
                cmap = plt.cm.binary if binary else plt.cm.gray
            )

            dst_file = os.path.join(self.dst_dir, file_name + '_' + str(i + 1).zfill(Preprocessor.FILL_DIGITS) + '.' + file_suffix + Preprocessor.IMAGE_FORMAT)
            fig.savefig(dst_file, frameon = True, transparent = False)

            Image.open(dst_file).convert('1' if binary else 'L').save(dst_file)
        end = time.clock()
        print('Saved all slices in %.2f seconds.' % (end - start))

        plt.close(fig)

    def _split(self):
        files = []
        for file in os.listdir(self.dst_dir):
            if os.path.isdir(file):
                continue

            file_name, file_extension = os.path.splitext(file)
            if not file_name.endswith(Preprocessor.INPUT_SUFFIX):
                continue

            input_file_name = file_name
            input_file = file
            output_file_name = os.path.splitext(file_name)[0] + '.' + Preprocessor.OUTPUT_SUFFIX
            output_file = output_file_name + file_extension
            if os.path.isfile(os.path.join(self.dst_dir, output_file)):
                files.append((os.path.join(self.dst_dir, input_file), os.path.join(self.dst_dir, output_file)))

        dataset_size = len(files)
        training_dataset_size = int(0.8*dataset_size)
        test_dataset_size = int(0.2*dataset_size)
        rounding_error = dataset_size - training_dataset_size - test_dataset_size
        training_dataset_size = training_dataset_size + rounding_error

        np.random.shuffle(files)
        training_files, test_files = files[:training_dataset_size], files[training_dataset_size:]

        for f in training_files:
            input_file, output_file = f
            shutil.move(input_file, self.training_dir)
            shutil.move(output_file, self.training_dir)
        for f in test_files:
            input_file, output_file = f
            shutil.move(input_file, self.test_dir)
            shutil.move(output_file, self.test_dir)
