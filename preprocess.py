'''
created: 2018-06-14
edited: 2018-07-03
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

DATA_SRC_PATH = './data/raw/MIREX/'
IMAGE_FORMAT = '.png'

# TODO: debug
#       make the size of the output/input pair images a parameter
#       implement q-transform as an alternative to SFFT
#       use logging instead of prints

def grey_scale(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

def binarize(img, threshold):
    img = grey_scale(img)
    img = img.astype(np.uint8)
    return np.where(img > threshold, 255, 0)

class Preprocessor:
    DATA_DEST_PATH = './data/preprocessed/MIREX/'
    DOWNSAMPLE_RATE = 8192 #Hz
    SFFT_WINDOW_LENGTH = 1024
    SFFT_STRIDE = 768 #SFFT_WINDOW_LENGTH//2

    @staticmethod
    def calc_rounded_slice_length(num, dem):
        '''
            We calculate the closest that would give a remainder of 0.
            This way we avoid weird results when slicing the spectrogram.
            We will end up with N or N - 1 slices (where N is the number of slices if num%dem == 0).
        '''
        remainder = num%dem
        remainder = dem - remainder
        true_len = num + remainder
        return true_len//dem

    def preprocess(self, dir_path, gen_bmp = False, plot = False, save_full = False):
        self._create_data_dirs()

        # TODO break this function up with auxiliary ones
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
            if save_full:
                self._save_spectrogram(spectrogram, file_name + IMAGE_FORMAT)
            if plot:
                spectrogram.plot()

            slice_length = Preprocessor.calc_rounded_slice_length(len(spectrogram.times), AudioReader.calc_signal_length_seconds(samples, sample_rate))
            self._save_sliced_spectrogram(spectrogram, file_name, slice_length)

            print()
        
        # Generate visualization slices for all mid files (output)
        for file in os.listdir(dir_path):
            file_name, file_extension = os.path.splitext(file)
            if file_extension != '.mid':
                continue

            print('Generating piano roll for %s.' % file)

            mid = MidiFile(os.path.join(dir_path, file), verbose = False)
            if save_full:
                mid.save_roll(os.path.join(Preprocessor.DATA_DEST_PATH, file_name + IMAGE_FORMAT))
            if plot:
                mid.draw_roll(draw_colorbar = False)

            slice_length = Preprocessor.calc_rounded_slice_length(np.shape(mid.get_roll_image())[1], mid.length_seconds)
            self._save_sliced_piano_roll(mid, file_name, slice_length)

            print()

        # Transform into BMP to get rid of alpha channel
        if gen_bmp:
            for file in os.listdir(Preprocessor.DATA_DEST_PATH):
                file_name, _ = os.path.splitext(file)
                img = Image.open(os.path.join(Preprocessor.DATA_DEST_PATH, file))
                out_file = os.path.join(Preprocessor.DATA_DEST_PATH, file_name + '.bmp')
                img.save(out_file)

    def one_hot_encode(self):
        for file in os.listdir(Preprocessor.DATA_DEST_PATH):
            file_name, file_extension = os.path.splitext(file)
            if file_extension == '.bmp':
                if file_name[-1] == 'n':
                    continue

                img = Image.open(os.path.join(Preprocessor.DATA_DEST_PATH, file))
                img = np.array(img)
                for row in img:
                    for col in row:
                        if col[0] == 255:
                            col[0] = col[1] = col[2] = 1

                img = Image.fromarray(img)
                out_file = os.path.join(Preprocessor.DATA_DEST_PATH, file_name + '_hot.bmp')
                img.save(out_file)

    def _create_data_dirs(self):
        if not os.path.exists(Preprocessor.DATA_DEST_PATH):
            os.makedirs(Preprocessor.DATA_DEST_PATH)

    def _save_sliced_spectrogram(self, spectrogram, file_name, slice_length):
        print(np.shape(spectrogram.values)[1]//slice_length)

        fig, ax = plt.subplots(1, figsize = (4, 16), dpi = 32)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')
    
        start = time.clock()
        chunks = spectrogram.get_chunk_generator(slice_length)
        for i, c in enumerate(chunks):
            # REMOVE
            if i > 48:
                break
            # Don't save the last slice if it has a different size from the rest
            if np.shape(c)[1] < slice_length:
                break
            ax.clear()
            ax.pcolormesh(
                spectrogram.times[i*slice_length:i*slice_length + slice_length],
                spectrogram.frequencies,
                c
            )
            dest_path = os.path.join(Preprocessor.DATA_DEST_PATH, file_name + '_' + str(i).zfill(3) + '_in' + IMAGE_FORMAT)
            fig.savefig(dest_path, frameon = True)
        end = time.clock()
        print('Saved all spectrogram slices in %.2f seconds.' % (end - start))

        plt.close(fig)

    def _save_sliced_piano_roll(self, mid, file_name, slice_length):
        fig, ax = plt.subplots(1, figsize = (4, 16), dpi = 32)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)

        start = time.clock()
        chunks = mid.get_chunk_generator(slice_length)
        for i, c in enumerate(chunks):
            # REMOVE
            if i > 48:
                break
            # Don't save the last slice if it has a different size from the rest
            if np.shape(c)[1] < slice_length:
                break
                
            ax.clear()
            ax.axis('off')

            c = binarize(c, 200)
            ax.imshow(c, aspect = 'auto', cmap = plt.cm.gray)

            dest_path = os.path.join(Preprocessor.DATA_DEST_PATH, file_name + '_' + str(i).zfill(3) + '_out' + IMAGE_FORMAT)
            fig.savefig(dest_path, frameon = True)
        end = time.clock()
        print('Saved all piano roll slices in %.2f seconds.' % (end - start))

        plt.close(fig)

    def _save_spectrogram(self, spectrogram, file_name):
        fig, ax = plt.subplots(1, figsize = (4, 16), dpi = 32)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')
        ax.pcolormesh(spectrogram.times, spectrogram.frequencies, spectrogram.values)
        dest_path = os.path.join(Preprocessor.DATA_DEST_PATH, file_name)
        fig.savefig(dest_path, frameon = True)
        plt.close(fig)

def main():
    preprocessor = Preprocessor()
    preprocessor.preprocess(DATA_SRC_PATH, gen_bmp = False, plot = False)

if __name__ == '__main__':
    main()
