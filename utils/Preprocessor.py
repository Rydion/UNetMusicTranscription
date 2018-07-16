'''
created: 2018-07-16
edited: 2018-07-16
author: Adrian Hintze @Rydion
'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from midiviz.midiviz import MidiFile
from utils.AudioReader import AudioReader
from utils.Spectrogram import Spectrogram
from utils.CQTransform import CQTransform
from utils.functions import grey_scale, binarize

# TODO:
#   implement q-transform as an alternative to SFFT
#   support other extensions apart from wav
#   make the size of the output/input pair images a parameter
#   use logging instead of prints

class Preprocessor:
    IMAGE_FORMAT = '.png'
    DOWNSAMPLE_RATE = 8192 #Hz
    SFFT_WINDOW_LENGTH = 1024
    SFFT_STRIDE = 768
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

    def preprocess(self, src_dir, dst_dir, transformation = 'spectrogram'):
        self._create_dst_dir(dst_dir)

        # TODO break this function up into auxiliary ones

        # Generate spectrogram slices for all wav files (input)
        for file in os.listdir(src_dir):
            file_name, file_extension = os.path.splitext(file)
            if file_extension != '.wav':
                continue

            print('Generating spectrogram for %s.' % file)

            audioreader = AudioReader()
            sample_rate, samples = audioreader.read_wav(
                os.path.join(src_dir, file),
                as_mono = True,
                downsample = True,
                downsample_rate = Preprocessor.DOWNSAMPLE_RATE
            )

            if transformation == 'spectrogram':
                spectrum = Spectrogram.from_audio(
                    sample_rate,
                    samples,
                    window_length = Preprocessor.SFFT_WINDOW_LENGTH,
                    stride = Preprocessor.SFFT_STRIDE
                )
            elif transformation == 'cqt':
                spectrum = CQTransform.from_audio(
                    sample_rate,
                    samples,
                    window_length = Preprocessor.SFFT_WINDOW_LENGTH,
                    stride = Preprocessor.SFFT_STRIDE
                )
            else:
                raise ValueError('Unknown transformation: ' + transformation + '.')

            slice_length = Preprocessor.calc_rounded_slice_length(np.shape(spectrum.values)[1], AudioReader.calc_signal_length_seconds(samples, sample_rate))
            self._save_sliced_spectrogram(spectrum, dst_dir, file_name, slice_length)

            # Output aesthetics
            print()
        
        # Generate visualization slices for all mid files (output)
        for file in os.listdir(src_dir):
            file_name, file_extension = os.path.splitext(file)
            if file_extension != '.mid':
                continue

            print('Generating piano roll for %s.' % file)

            mid = MidiFile(os.path.join(src_dir, file), verbose = False)
            slice_length = Preprocessor.calc_rounded_slice_length(np.shape(mid.get_roll_image())[1], mid.length_seconds)
            self._save_sliced_piano_roll(mid, dst_dir, file_name, slice_length)

            # Output aesthetics
            print()

    def _create_dst_dir(self, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

    # TODO merge code for _save_sliced_*
    def _save_sliced_spectrogram(self, spectrum, dst_dir, file_name, slice_length):
        fig, ax = plt.subplots(1, figsize = (4, 16), dpi = 32)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
    
        start = time.clock()
        chunks = spectrum.get_chunk_generator(slice_length)
        for i, c in enumerate(chunks):
            if i >= 48:
                break
            # Don't save the last slice if it is smaller than the rest
            if np.shape(c)[1] < slice_length:
                break

            ax.clear()
            ax.axis('off')
            ax.imshow(c, aspect = 'auto', cmap = plt.cm.gray)

            dst_file = os.path.join(dst_dir, file_name + '_' + str(i).zfill(Preprocessor.FILL_DIGITS) + '_in' + Preprocessor.IMAGE_FORMAT)
            fig.savefig(dst_file, frameon = True, transparent = False)

            Image.open(dst_file).convert('L').save(dst_file)
        end = time.clock()
        print('Saved all spectrogram slices in %.2f seconds.' % (end - start))

        plt.close(fig)

    def _save_sliced_piano_roll(self, mid, dst_dir, file_name, slice_length):
        fig, ax = plt.subplots(1, figsize = (4, 16), dpi = 32)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)

        start = time.clock()
        chunks = mid.get_chunk_generator(slice_length)
        for i, c in enumerate(chunks):
            if i >= 48:
                break
            # Don't save the last slice if it is smaller than the rest
            if np.shape(c)[1] < slice_length:
                break

            c = grey_scale(c)
            c = binarize(c, 200)

            ax.clear()
            ax.axis('off')
            ax.imshow(c, aspect = 'auto', cmap = plt.cm.binary)

            dst_file = os.path.join(dst_dir, file_name + '_' + str(i).zfill(Preprocessor.FILL_DIGITS) + '_out' + Preprocessor.IMAGE_FORMAT)
            fig.savefig(dst_file, frameon = True, transparent = False)

            Image.open(dst_file).convert('1').save(dst_file)
        end = time.clock()
        print('Saved all piano roll slices in %.2f seconds.' % (end - start))

        plt.close(fig)
