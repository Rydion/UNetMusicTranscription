'''
created: 2018-06-15
edited: 2018-07-16
author: Adrian Hintze @Rydion
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from utils.Spectrum import Spectrum
from utils.functions import normalize_array

class Spectrogram(Spectrum):
    @classmethod
    def from_audio(cls, sample_rate, samples, window_length = 1024, stride = 512):
        frequencies, times, values = signal.spectrogram(
            samples,
            fs = sample_rate,
            nperseg = window_length,
            noverlap = stride,
            mode = 'magnitude'
        )
        print(np.shape(values)[0])
        values = 20*np.log10(values)
        values = Spectrogram.normalize_values(values)
        return Spectrogram(frequencies, times, values)

    def __init__(self, frequencies, times, values):
        super().__init__(values)
        self._frequencies = frequencies
        self._times = times

    def plot(self):
        plt.pcolormesh(self.times, self.frequencies, self.values)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def times(self):
        return self._times
