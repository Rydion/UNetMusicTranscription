'''
created on 2018-06-14
author: Adrian Hintze @Rydion
'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from src.midi import roll
from src.utils.AudioReader import AudioReader
from src.utils.Spectrogram import Spectrogram

DATA_SRC_PATH = './data/raw/MIREX/'

# TODO: debug
#       make the size of the output/input pair images a parameter
#       implement midi to piano roll conversion
#       implement q-transform as an alternative to SFFT
#       use logging instead of prints


class Preprocessor:
    INPUT_DATA_DEST_PATH = './data/preprocessed/MIREX/input'
    OUTPUT_DATA_DEST_PATH = './data/preprocessed/MIREX/output'
    DOWNSAMPLE_RATE = 8192 #Hz
    SFFT_WINDOW_LENGTH = 1024
    SFFT_STRIDE = 768 #SFFT_WINDOW_LENGTH//2

    def preprocess(self, dir_path, plot = False):
        # TODO break this function up with auxiliary ones
        # Generate spectrogram slices for all wav files (input)
        for file in os.listdir(dir_path):
            break
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
        
        # Generate visualization slices for all mid files (output)
        for file in os.listdir(dir_path):
            file_name, file_extension = os.path.splitext(file)
            if file_extension != '.mid':
                continue

            mid = roll.MidiFile(os.path.join(dir_path, file), verbose = False)
            mid.save_roll(os.path.join(Preprocessor.OUTPUT_DATA_DEST_PATH, file_name + '.png'))
            mid.draw_roll(draw_colorbar = False)

            fig, ax = plt.subplots(1, figsize = (4, 16), dpi = 32)
            fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)

            img = mid.get_roll_image()
            slice_length = np.shape(img)[1]//54
            chunks = mid.get_chunk_generator(slice_length)
            for i, c in enumerate(chunks):
                if np.shape(c)[1] < slice_length:
                    break
                
                ax.clear()
                ax.axis('off')
                ax.imshow(c, interpolation='nearest', aspect='auto')

                dest_path = os.path.join(Preprocessor.OUTPUT_DATA_DEST_PATH, file_name + '_' + str(i).zfill(3) + '.png')
                fig.savefig(dest_path)

            plt.close(fig)
            break
        
    def _save_sliced_spectrogram(self, spectrogram, file_name, slice_length):
        # TODO move this somewhere more appropiate
        if not os.path.exists(Preprocessor.INPUT_DATA_DEST_PATH):
            os.makedirs(Preprocessor.INPUT_DATA_DEST_PATH)

        fig, ax = plt.subplots(1, figsize = (4, 16), dpi = 32)
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
            fig.savefig(dest_path)
        end = time.clock()
        print('Saved all slices in %.2f seconds.' % (end - start))

        plt.close(fig)

    def _save_spectrogram(self, spectrogram, file_name):
        fig, ax = plt.subplots(1, figsize = (4, 16), dpi = 32)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')
        ax.pcolormesh(spectrogram.times, spectrogram.frequencies, spectrogram.values)
        dest_path = os.path.join(Preprocessor.INPUT_DATA_DEST_PATH, file_name + '.png')
        fig.savefig(dest_path)
        plt.close(fig)

def main():
    preprocessor = Preprocessor()
    preprocessor.preprocess(DATA_SRC_PATH)

if __name__ == '__main__':
    main()
