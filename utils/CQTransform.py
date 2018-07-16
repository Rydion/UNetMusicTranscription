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

class CQTransform(Spectrum):
    @classmethod
    def from_audio(cls, sample_rate, samples, window_length = 1024, stride = 512):
        values = librosa.core.cqt(
            samples,
            sr = sample_rate,
            hop_length = stride,
            n_bins = 82
        )
        values = librosa.amplitude_to_db(abs(values))
        return CQTransform(values)

    @staticmethod
    def normalize_values(values):
        return normalize_array(values)

    def __init__(self, values):
        super().__init__(values)

    def plot(self):
        plt.pcolormesh(self.values, aspect = 'auto', cmap = plt.cm.hot)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    def get_chunk_generator(self, chunk_length):
        for i in range(0, np.shape(self.values)[1], chunk_length):
            yield self.values[:, i:i + chunk_length]

    @property
    def values(self):
        return self._values
