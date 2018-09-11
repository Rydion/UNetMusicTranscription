'''
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
        multitrack.trim_trailing_silence()
        multitrack.remove_empty_tracks()
        multitrack.pad_to_same()
        if len(multitrack.tracks) > 10000:
            multitrack.merge_tracks(
                mode = 'max',
                is_drum = False,
                program = 0,
                remove_merged = True
            )

        return Midi(multitrack)

    def __init__(self, multitrack):
        self._multitrack = multitrack
        self._trim()

    def get_length_seconds(self):
        resolution = self._multitrack.beat_resolution
        seconds = 0
        for i in range(self._multitrack.get_maximal_length()):
            tempo = self._multitrack.tempo[i]
            seconds = seconds + 60/(tempo*resolution)
        return seconds
        '''
        tempo = self._multitrack.tempo[0] # Supose constant tempo
        total_ticks = self._multitrack.get_maximal_length()
        resolution = self._multitrack.beat_resolution

        print(['test', secs, (60*total_ticks)/(tempo*resolution)])

        return (60*total_ticks)/(tempo*resolution)
        '''

    def plot(self, x, plain = False):
        fig, ax = self._plot(x, 1, plain = plain)
        plt.show(fig)
        plt.close(fig)

    def save(self, dest_path, x, plain = False):
        fig, ax = self._plot(x, 1, plain = plain)
        if plain:
            fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        fig.savefig(dest_path)
        plt.close(fig)

    def get_img(self, x, plain = True):
        fig, ax = self._plot(x, 1, plain = plain)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype = 'uint8')

        width, height = fig.get_size_inches()*fig.get_dpi()
        width = int(width)
        height = int(height)
        img = img.reshape(height, width, 3)

        plt.close(fig)

        return img

    def _plot(self, mult_x, mult_y, plain):
        default_figsize = plt.rcParams['figure.figsize']
        plt.rcParams['figure.figsize'] = [mult_x*default_figsize[0], default_figsize[1]]
        fig, ax = pyano.plot(
            self._multitrack,
            preset = 'plain' if plain else 'default',
            ytick = 'pitch',
            xtick = 'step'
        )
        plt.rcParams['figure.figsize'] = default_figsize

        return fig, ax

    def _trim(self):
        duration = self.get_length_seconds()
        duration = int(duration)
        total_ticks = self._get_ticks_from_seconds(duration)
        for i, t in enumerate(self._multitrack.tracks):
            self._multitrack.tracks[i] = t[:total_ticks]

    def _get_ticks_from_seconds(self, duration):
        s = set()
        for i in self._multitrack.tempo:
            s.add(i)
        print(s)

        tempo = self._multitrack.tempo[0] # Supose constant tempo
        total_ticks = self._multitrack.get_maximal_length()
        resolution = self._multitrack.beat_resolution
        return int(duration*(tempo*resolution)//60)
