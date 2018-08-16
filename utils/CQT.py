'''
created: 2018-07-16
edited: 2018-07-16
author: Adrian Hintze @Rydion
'''

import librosa
import numpy as np
import matplotlib.pyplot as plt

from utils.Spectrum import Spectrum
from utils.functions import normalize_array

class CQT(Spectrum):
    @classmethod
    def from_audio(cls, sample_rate, samples, stride = 512):
        values = librosa.core.cqt(
            samples,
            sr = sample_rate,
            #n_bins = 7,
            #bins_per_octave = 36,
            #hop_length = stride
        )
        values = librosa.amplitude_to_db(np.abs(values), ref = np.max)
        values = CQT.normalize_values(values)
        return CQT(values, sample_rate)

    def __init__(self, *args):
        super().__init__(*args)

    def plot(self, color = True):
        fig, ax = plt.subplots(1)
        librosa.display.specshow(
            self.values,
            sr = self.sample_rate,
            y_axis = 'cqt_hz',
            ax = ax,
            cmap = None if color else plt.cm.gray
        )
        fig.title('Constant-Q power spectrogram (HZ)')
        plt.show(fig)
        plt.close(fig)

    def save(self, dest_path, color = True):
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')

        librosa.display.specshow(
            self.values,
            y_axis = 'cqt_hz',
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
            y_axis = 'cqt_hz',
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
