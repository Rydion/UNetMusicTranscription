'''
author: Adrian Hintze @Rydion
'''

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from functions import normalize_array

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

    def plot(self, x, y, color = True):
        fig, ax = self._plot(x, y, color)
        plt.show(fig)
        plt.close(fig)

    def save(self, dst_path, x, y, color = True):
        fig, ax = self._plot(x, y, color)
        fig.savefig(dst_path)
        plt.close(fig)

    def get_img(self, x, y, color = True):
        fig, ax = self._plot(x, y, color)
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

    @abstractmethod
    def _plot(self, x, y, color):
        pass

    @property
    def values(self):
        return self._values

    @property
    def sample_rate(self):
        return self._sample_rate
