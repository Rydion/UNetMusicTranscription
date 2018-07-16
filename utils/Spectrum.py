'''
created: 2018-06-15
edited: 2018-07-16
author: Adrian Hintze @Rydion
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from abc import ABC, abstractmethod
from utils.functions import normalize_array

class Spectrum(ABC):
    @staticmethod
    def normalize_values(values):
        return normalize_array(values)

    @classmethod
    @abstractmethod
    def from_audio(cls, sample_rate, samples, window_length = 1024, stride = 512):
        pass

    def __init__(self, values):
        self._values = values

    @abstractmethod
    def plot(self):
        pass

    def get_chunk_generator(self, chunk_length):
        for i in range(0, np.shape(self.values)[1], chunk_length):
            yield self.values[:, i:i + chunk_length]

    @property
    def values(self):
        return self._values
