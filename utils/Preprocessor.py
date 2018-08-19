'''
created: 2018-07-16
author: Adrian Hintze @Rydion
'''

import os
import time
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
#   use logging instead of prints

class Preprocessor:
    IMAGE_FORMAT = '.png'
    DOWNSAMPLE_RATE = 16000 #Hz
    SFFT_WINDOW_LENGTH = 1024
    SFFT_STRIDE = SFFT_WINDOW_LENGTH//2
    FILL_DIGITS = 4

    @staticmethod
    def calc_rounded_slice_length(num, dem):
        """
            We calculate the closest that would give a remainder of 0.
            This way we avoid weird results when slicing the spectrogram.
            We will end up with N or N - 1 slices (where N is the number of slices if num%dem == 0).
        """
        remainder = num%dem
        remainder = dem - remainder
        true_len = num + remainder
        return true_len//dem

    def preprocess(self, src_dir, dst_dir, gen_input = True, gen_output = True, transformation = 'spectrogram'):
        self._create_dst_dir(dst_dir)

        for file in os.listdir(src_dir):
            file_name, file_extension = os.path.splitext(file)
            # Search for audio files
            if file_extension != '.wav':
                continue

            # Skip any input file that does not have a corresponding output
            if not os.path.isfile(os.path.join(src_dir, file_name + '.mid')):
                continue

            print('Generating input/output for %s.' % file_name)

            # Read original files
            sample_rate, samples = AudioReader.read_wav(
                os.path.join(src_dir, file),
                as_mono = True,
                downsample = True,
                downsample_rate = Preprocessor.DOWNSAMPLE_RATE
            )

            midi = Midi.from_file(os.path.join(src_dir, file_name + '.mid'))

            # If input and output have different lengths we still want to use the data
            duration = min(midi.get_length_seconds(), AudioReader.calc_signal_length_seconds(samples, sample_rate))

            if duration < AudioReader.calc_signal_length_seconds(samples, sample_rate):
                sample_rate, samples = AudioReader.read_wav(
                    os.path.join(src_dir, file),
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

            subdivisions = int(duration)

            if gen_input:
                spectrogram.save(os.path.join(dst_dir, file_name + '.spectrum.png'), subdivisions, 84, color = False)
                spectrogram_img = spectrogram.get_img(subdivisions, 84)
                slice_length = Preprocessor.calc_rounded_slice_length(np.shape(spectrogram_img)[1], subdivisions)
                chunks = get_chunk_generator(spectrogram_img, slice_length)
                self._save_sliced(chunks, dst_dir, file_name, file_suffix = 'in', binary = False)

            if gen_output:
                midi.save(os.path.join(dst_dir, file_name + '.mid.png'), subdivisions)
                midi_img = midi.get_img(subdivisions)
                slice_length = Preprocessor.calc_rounded_slice_length(np.shape(midi_img)[1], subdivisions)
                chunks = get_chunk_generator(midi_img, slice_length)
                self._save_sliced(chunks, dst_dir, file_name, file_suffix = 'out', binary = True)

            # Output aesthetics
            print()

    def _create_dst_dir(self, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

    def _save_sliced(self, chunks, dst_dir, file_name, file_suffix = '', binary = False):
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
                #c = c[200:320, :]

            ax.clear()
            ax.axis('off')
            ax.imshow(
                c,
                aspect = 'auto',
                vmin = 0,
                vmax = 255 if binary else 1,
                cmap = plt.cm.binary if binary else plt.cm.gray
            )

            dst_file = os.path.join(dst_dir, file_name + '_' + str(i + 1).zfill(Preprocessor.FILL_DIGITS) + '.' + file_suffix + Preprocessor.IMAGE_FORMAT)
            fig.savefig(dst_file, frameon = True, transparent = False)

            Image.open(dst_file).convert('1' if binary else 'L').save(dst_file)
        end = time.clock()
        print('Saved all slices in %.2f seconds.' % (end - start))

        plt.close(fig)
