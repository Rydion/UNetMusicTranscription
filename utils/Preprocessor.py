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
#   support other extensions apart from wav
#   make the size of the output/input pair images a parameter
#   use logging instead of prints

class Preprocessor:
    IMAGE_FORMAT = '.png'
    DOWNSAMPLE_RATE = 8192 #Hz
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
            if file_name != 'h':
                #continue
                pass
            if file_extension != '.wav':
                continue

            # Skip any input file that does not have a corresponding output
            if not os.path.isfile(os.path.join(src_dir, file_name + '.mid')):
                continue

            print('Generating input/output for %s.' % file_name)

            # Read original files
            audioreader = AudioReader()
            sample_rate, samples = audioreader.read_wav(
                os.path.join(src_dir, file),
                as_mono = True,
                downsample = True,
                downsample_rate = Preprocessor.DOWNSAMPLE_RATE
            )

            mid = MidiFile(
                os.path.join(src_dir, file_name + '.mid'),
                sr = 1, # TODO calculate to make the result multiple of length
                verbose = False
            )
            print(mid.total_ticks)

            # If input and output have different lengths we still want to use the data
            length = min(mid.length_seconds, AudioReader.calc_signal_length_seconds(samples, sample_rate))

            # Crop sound and mid to the nearest second integer
            samples = samples[0:sample_rate*length]
            # TODO crop mid

            # Crop spectrum/mid to length
            # Calculate slice length
            # create slices

            if transformation == 'spectrogram':
                spectrum = Spectrogram.from_audio(
                    sample_rate,
                    samples,
                    window_length = Preprocessor.SFFT_WINDOW_LENGTH,
                    stride = Preprocessor.SFFT_WINDOW_LENGTH - Preprocessor.SFFT_STRIDE
                )
            elif transformation == 'cqt':
                spectrum = CQTransform.from_audio(
                    sample_rate,
                    samples,
                    stride = Preprocessor.SFFT_STRIDE
                )
            else:
                raise ValueError('Unknown transformation: ' + transformation + '.')

            if gen_input:
                spectrum.save(os.path.join(dst_dir, file_name + '.spectrum.png'))
                #slice_length = Preprocessor.calc_rounded_slice_length(np.shape(spectrum.values)[1], length)
                slice_length = Preprocessor.calc_rounded_slice_length(np.shape(spectrum.get_img())[1], length)
                chunks = spectrum.get_chunk_generator(slice_length)
                self._save_sliced(chunks, dst_dir, file_name, file_suffix = 'in', binary = False)

            if gen_output:
                mid.save_roll(os.path.join(dst_dir, file_name + '.mid.png'))
                slice_length = Preprocessor.calc_rounded_slice_length(np.shape(mid.get_roll_image())[1], length)
                chunks = mid.get_chunk_generator(slice_length)
                self._save_sliced(chunks, dst_dir, file_name, file_suffix = 'out', binary = True)

            # Output aesthetics
            print()
            #break

    def _create_dst_dir(self, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

    def _save_sliced(self, chunks, dst_dir, file_name, file_suffix = '', binary = False):
        #fig, ax = plt.subplots(1, figsize = (4, 11), dpi = 16)
        fig, ax = plt.subplots(1, figsize = (4, 16), dpi = 32)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)

        start = time.clock()
        slice_length = 0
        for i, c in enumerate(chunks):
            if i >= 40: # TODO
                #break
                pass

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

            dst_file = os.path.join(dst_dir, file_name + '_' + str(i + 1).zfill(Preprocessor.FILL_DIGITS) + '_' + file_suffix + Preprocessor.IMAGE_FORMAT)
            fig.savefig(dst_file, frameon = True, transparent = False)

            Image.open(dst_file).convert('1' if binary else 'L').save(dst_file)
        end = time.clock()
        print('Saved all slices in %.2f seconds.' % (end - start))

        plt.close(fig)
