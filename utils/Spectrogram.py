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

class Spectrogram(ABC):
    @staticmethod
    def normalize_values(values):
        return normalize_array(values)

    @classmethod
    @abstractmethod
    def from_audio(cls, sample_rate, samples, window_length = 1024, stride = 512):
        pass

    def __init__(self, values, sample_rate):
        self._values = values
        self._sample_rate = sample_rate

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def get_img(self):
        pass

    @property
    def values(self):
        return self._values

    @property
    def sample_rate(self):
        return self._sample_rate
