'''
created on 2018-06-14
author: Adrian Hintze @Rydion
'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from src.utils.AudioReader import AudioReader
from src.utils.Spectrogram import Spectrogram

DATA_SRC_PATH = './data/raw/MIREX/'

# TODO: debug
#       implement midi to piano roll preprocessing
#       implement q-transform as an alternative to SFFT
#       use logging instead of prints


class Preprocessor:
    INPUT_DATA_DEST_PATH = './data/preprocessed/MIREX/input'
    OUTPUT_DATA_DEST_PATH = './data/preprocessed/MIREX/output'
    DOWNSAMPLE_RATE = 8192 #Hz
    SFFT_WINDOW_LENGTH = 1024
    SFFT_STRIDE = 768 #SFFT_WINDOW_LENGTH//2

    def preprocess(self, dir_path, plot = False):
        # Generate spectrogram slices for all wav files (input)
        for file in os.listdir(dir_path):
            file_name, file_extension = os.path.splitext(file)
            if file_extension != '.wav':
                continue

            print('Generating spectrogram for %s.' % file)

            audioreader = AudioReader()
            sample_rate, samples = audioreader.read_wav(
                os.path.join(dir_path, file),
                as_mono = True,
                downsample = True,
                downsample_rate = Preprocessor.DOWNSAMPLE_RATE
            )

            spectrogram = Spectrogram.from_audio(
                sample_rate,
                samples,
                window_length = Preprocessor.SFFT_WINDOW_LENGTH,
                stride = Preprocessor.SFFT_STRIDE
            )
            if plot:
                spectrogram.plot()

            slice_length = len(spectrogram.times)//AudioReader.calc_signal_length_seconds(samples, sample_rate)
            self._save_sliced_spectrogram(spectrogram, file_name, slice_length)

            print()

    def _save_sliced_spectrogram(self, spectrogram, file_name, slice_length):
        # TODO move this somewhere more appropiate
        if not os.path.exists(Preprocessor.INPUT_DATA_DEST_PATH):
            os.makedirs(Preprocessor.INPUT_DATA_DEST_PATH)

        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')
    
        start = time.clock()
        chunks = spectrogram.get_chunk_generator(slice_length)
        for i, c in enumerate(chunks):    
            # Don't save the last slice if it has a different size from the rest
            if np.shape(c)[1] < slice_length:
                break
            ax.clear()
            ax.pcolormesh(
                spectrogram.times[i*slice_length:i*slice_length + slice_length],
                spectrogram.frequencies,
                c
            )
            dest_path = os.path.join(Preprocessor.INPUT_DATA_DEST_PATH, file_name + '_' + str(i).zfill(3) + '.png')
            fig.savefig(dest_path, dpi = 300, frameon = 'false')
        end = time.clock()
        print('Saved all slices in %.2f seconds.' % (end - start))

        plt.close(fig)

    def _save_spectrogram(self, spectrogram, file_name):
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')
        ax.pcolormesh(spectrogram.times, spectrogram.frequencies, spectrogram.values)
        fig.savefig(os.path.join(Preprocessor.INPUT_DATA_DEST_PATH, file_name + '.png'), dpi = 300, frameon = 'false')
        plt.close(fig)

def main():
    preprocessor = Preprocessor()
    preprocessor.preprocess(DATA_SRC_PATH)

if __name__ == '__main__':
    main()
