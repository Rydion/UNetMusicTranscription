'''
created: 2018-06-15
author: Adrian Hintze @Rydion
'''

import numpy as np
import matplotlib.pyplot as plt

import pypianoroll as pyano

class Midi:
    @staticmethod
    def from_file(file_path):
        multitrack = pyano.parse(file_path)
        multitrack.assign_constant(127)
        return Midi(multitrack)

    def __init__(self, multitrack):
        self._multitrack = multitrack

    def plot(self):
        fig, ax = pyano.plot(
            self._multitrack,
            ytick = 'pitch',
            xtick = 'step'
        )
        plt.show(fig)
        plt.close(fig)

    def save(self, dest_path):
        fig, ax = pyano.plot(
            self._multitrack,
            preset = 'plain',
            ytick = 'pitch',
            xtick = 'step'
        )
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        fig.savefig(dest_path)
        plt.close(fig)

    def get_img(self):
        fig, ax = pyano.plot(
            self._multitrack,
            preset = 'plain',
            ytick = 'pitch',
            xtick = 'step'
        )
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype = 'uint8')

        width, height = fig.get_size_inches()*fig.get_dpi()
        width = int(width)
        height = int(height)
        img = img.reshape(height, width, 3)

        plt.close(fig)

        return img

    def get_chunk_generator(self, chunk_length):
        img = self.get_img()
        for i in range(0, np.shape(img)[1], chunk_length):
            yield img[:, i:i + chunk_length]

    def get_length_seconds(self):
        tempo = self._multitrack.tempo[0] # Supose constant tempo
        total_ticks = self._multitrack.get_maximal_length()
        resolution = self._multitrack.beat_resolution
        return (60*total_ticks)//(tempo*resolution)

