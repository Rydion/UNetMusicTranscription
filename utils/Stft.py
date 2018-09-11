'''
author: Adrian Hintze @Rydion
'''

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from utils.Spectrogram import Spectrogram

class Stft(Spectrogram):
    @classmethod
    def from_audio(cls, sample_rate, samples, window_length = 1024, stride = 512):
        values = librosa.stft(
            samples,
            n_fft = window_length,
            hop_length = stride
        )
        values = librosa.amplitude_to_db(np.abs(values), ref = np.max)
        #values = Stft.normalize_values(values)
        return Stft(values, sample_rate)

    def __init__(self, *args):
        super().__init__(*args)

    def _plot(self, mult_x, mult_y, color):
        figsize_x, figsize_y = plt.rcParams['figure.figsize']
        fig, ax = plt.subplots(1, figsize = (mult_x*figsize_x, figsize_y))
        librosa.display.specshow(
            self.values,
            sr = self.sample_rate,
            y_axis = 'log',
            x_axis = 'time',
            ax = ax,
            cmap = None if color else plt.cm.gray
        )
        return fig, ax
