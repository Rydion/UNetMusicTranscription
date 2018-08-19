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

    def get_length_seconds(self):
        tempo = self._multitrack.tempo[0] # Supose constant tempo
        total_ticks = self._multitrack.get_maximal_length()
        resolution = self._multitrack.beat_resolution
        return (60*total_ticks)//(tempo*resolution)

    def plot(self, x, plain = False):
        fig, ax = self._plot(x, plain = plain)
        plt.show(fig)
        plt.close(fig)

    def save(self, dest_path, x, plain = False):
        fig, ax = self._plot(x, plain = plain)
        if plain:
            fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        fig.savefig(dest_path)
        plt.close(fig)

    def get_img(self, x, plain = True):
        fig, ax = self._plot(x, plain = plain)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype = 'uint8')

        width, height = fig.get_size_inches()*fig.get_dpi()
        width = int(width)
        height = int(height)
        img = img.reshape(height, width, 3)

        plt.close(fig)

        return img

    def _plot(self, x, y = 1, plain = True):
        default = plt.rcParams['figure.figsize']
        plt.rcParams['figure.figsize'] = [x*4, y*16]
        fig, ax = pyano.plot(
            self._multitrack,
            preset = 'plain' if plain else 'default',
            ytick = 'pitch',
            xtick = 'step'
        )
        plt.rcParams['figure.figsize'] = default

        return fig, ax
