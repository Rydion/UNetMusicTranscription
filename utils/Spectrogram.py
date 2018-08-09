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
            window = signal.get_window('hann', window_length),
            noverlap = stride,
            #scaling = 'spectrum',
            #mode = 'magnitude'
        )
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

    def save(self, dest_path):
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')

        ax.imshow(
            self.values,
            aspect = 'auto',
            cmap = plt.cm.gray
        )
        fig.savefig(dest_path)
        
        plt.close(fig)

    def get_chunk_generator(self, chunk_length):
        img = self.get_img()
        for i in range(0, np.shape(img)[1], chunk_length):
            yield img[:, i:i + chunk_length]

    def get_img(self):
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')
        ax.set_facecolor('black')

        ax.imshow(
            self.values,
            aspect = 'auto',
            cmap = plt.cm.gray
        )

        #canvas = FigureCanvasAgg(fig)
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
