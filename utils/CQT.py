'''
created: 2018-07-16
edited: 2018-07-16
author: Adrian Hintze @Rydion
'''

import librosa
import numpy as np
import matplotlib.pyplot as plt

from utils.Spectrogram import Spectrogram
from utils.functions import normalize_array

class CQT(Spectrogram):
    @classmethod
    def from_audio(cls, sample_rate, samples, stride = 512):
        values = librosa.core.cqt(
            samples,
            sr = sample_rate,
            #n_bins = 7*12, # total num of bins
            #bins_per_octave = 12,
            #hop_length = stride
        )
        values = librosa.amplitude_to_db(np.abs(values), ref = np.max)
        #values = CQT.normalize_values(values)
        return CQT(values, sample_rate)

    def __init__(self, *args):
        super().__init__(*args)

    def plot(self, x, y, color = True):
        fig, ax = self._plot(x, y, color)
        plt.show(fig)
        plt.close(fig)

    def save(self, dest_path, x, y, color = True):
        fig, ax = self._plot(x, y, color)
        #fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        #ax.axis('off')

        fig.savefig(dest_path)

        plt.close(fig)

    def get_img(self, x, y):
        fig, ax = self._plot(x, y, color = False)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype = 'uint8')

        width, height = fig.get_size_inches()*fig.get_dpi()
        width = int(width)
        height = int(height)
        img = img.reshape(height, width, 3)

        plt.close(fig)

        return img

    def _plot(self, x, y, color = True):
        fig, ax = plt.subplots(1, figsize = (x*4, 16), dpi = 32)
        librosa.display.specshow(
            self.values,
            sr = self.sample_rate,
            y_axis = 'cqt_note',
            x_axis = 'time',
            ax = ax,
            cmap = None if color else plt.cm.gray
        )
        return fig, ax