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

        track_indices = [i for i in range(len(multitrack.tracks))]
        multitrack.merge_tracks(
            track_indices = track_indices,
            mode = 'max',
            is_drum = False,
            program = 0,
            remove_merged = True
        )
        
        return Midi(multitrack)

    def __init__(self, multitrack):
        self._multitrack = multitrack
        self._trim()

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

    def get_length_seconds(self):
        seconds = 0
        for i in range(self._multitrack.get_maximal_length()):
            tempo = self._multitrack.tempo[i]
            resolution = self._multitrack.beat_resolution
            seconds = seconds + 60/(tempo*resolution)
        return round(seconds, 2)

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
        print(self.get_length_seconds())
        for i in range(len(self._multitrack.tracks)):
            track = self._multitrack.tracks[i]
            print(total_ticks)
            print(np.shape(track.pianoroll))
            track.pianoroll = track.pianoroll[:total_ticks]
            print(np.shape(track.pianoroll))
            self._multitrack.tracks[i] = track
        print(self.get_length_seconds())

    def _get_ticks_from_seconds(self, duration):
        return int((duration*self._multitrack.tempo[0]*self._multitrack.beat_resolution)/60)
        '''
        ticks = 0
            for i in range(self._multitrack.get_maximal_length()):
                tempo = self._multitrack.tempo[i]
                resolution = self._multitrack.beat_resolution
                ticks = ticks + (duration/60)*(tempo*resolution)
            return int(ticks)
        '''
