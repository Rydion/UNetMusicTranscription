'''
created: 2018-06-15
author: Adrian Hintze @Rydion
'''

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from utils.Spectrum import Spectrum
from utils.functions import normalize_array

class Spectrogram(Spectrum):
    @classmethod
    def from_audio(cls, sample_rate, samples, window_length = 1024, stride = 512):
        values = librosa.stft(
            samples,
            n_fft = window_length,
            hop_length = stride
        )
        values = librosa.amplitude_to_db(np.abs(values), ref = np.max)
        values = Spectrogram.normalize_values(values)
        return Spectrogram(values)

    def __init__(self, values):
        super().__init__(values)

    def plot(self, color = True):
        fig, ax = plt.subplots(1)
        librosa.display.specshow(
            self.values,
            y_axis = 'log',
            ax = ax,
            cmap = None if color else plt.cm.gray
        )
        fig.title('Log-frequency power spectrogram')
        plt.show(fig)
        plt.close(fig)

    def save(self, dest_path, color = True):
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')

        librosa.display.specshow(
            self.values,
            y_axis = 'log',
            ax = ax,
            cmap = None if color else plt.cm.gray
        )

        fig.savefig(dest_path)

        plt.close(fig)

    def get_img(self):
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')

        librosa.display.specshow(
            self.values,
            y_axis = 'log',
            ax = ax,
            cmap = plt.cm.gray
        )

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype = 'uint8')

        width, height = fig.get_size_inches()*fig.get_dpi()
        width = int(width)
        height = int(height)
        img = img.reshape(height, width, 3)

        plt.close(fig)

        return img

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def times(self):
        return self._times
