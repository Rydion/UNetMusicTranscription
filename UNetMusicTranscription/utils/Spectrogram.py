'''
created on 2018-06-15
author: Adrian Hintze @Rydion
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from utils import utils

class Spectrogram:
    @staticmethod
    def from_audio(sample_rate, samples, window_length = 1024, stride = 512):
        frequencies, times, values = signal.spectrogram(
            samples,
            fs = sample_rate,
            nperseg = window_length,
            noverlap = stride,
            mode = 'magnitude'
        )
        values = 20*np.log10(values)
        values = Spectrogram.normalize_values(values)
        return Spectrogram(frequencies, times, values)

    @staticmethod
    def normalize_values(values):
        return utils.normalize_array(values)

    def __init__(self, frequencies, times, values):
        self._frequencies = frequencies
        self._times = times
        self._values = values

    def normalize(self):
        return self

    def plot(self):
        plt.pcolormesh(self.times, self.frequencies, self.values)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    def get_chunk_generator(self, chunk_length):
        for i in range(0, np.shape(self.values)[1], chunk_length):
            yield self.values[:, i:i + chunk_length]

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def times(self):
        return self._times

    @property
    def values(self):
        return self._values
