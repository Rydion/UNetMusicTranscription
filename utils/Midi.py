'''
author: Adrian Hintze
'''

import numpy as np
import matplotlib.pyplot as plt

import pretty_midi
import pypianoroll as pyano

# https://github.com/craffel/pretty-midi/issues/112
pretty_midi.pretty_midi.MAX_TICK = 1e10 # Pretty MIDI does some stuff internally that doesn't allow to load MAPS

class Midi:
    @staticmethod
    def from_file(file_path):
        multitrack = pyano.parse(file_path)

        multitrack.binarize()
        multitrack.assign_constant(1)

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

        sampled_pianoroll = self._sample(self._multitrack.tracks[0].pianoroll)
        sampled_pianoroll = sampled_pianoroll.transpose()
        print(np.shape(sampled_pianoroll))
        print([np.amin(sampled_pianoroll), np.amax(sampled_pianoroll)])

        pianoroll = self._multitrack.tracks[0].pianoroll#[:, 12:108]
        pianoroll = pianoroll.transpose()
        print(np.shape(pianoroll))
        print([np.amin(pianoroll), np.amax(pianoroll)])

        fig, ax = plt.subplots(1, 2, figsize = None, dpi = None)
        ax[0].imshow(pianoroll, vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
        ax[1].imshow(sampled_pianoroll, vmin = 0, vmax = 1, aspect = 'auto', cmap = plt.cm.gray)
        plt.draw()
        plt.show()

    def _sample(self, pianoroll, sps = 32):
        ticks = self._multitrack.get_maximal_length()
        length_seconds = self._get_length_seconds(ticks)
        time_step = 1/sps
        samples = int(length_seconds//time_step) + 1

        notes = 128
        sampled_pianoroll = np.zeros([samples, notes])

        def aux(ticks):
            time = 0
            for i in range(ticks):
                tempo = self._multitrack.tempo[i]
                resolution = self._multitrack.beat_resolution
                tick_time = 60/(tempo*resolution)
                time = time + tick_time
            return time

        i = 0
        time = 0
        current_tick = 0
        current_midi_time = aux(current_tick)
        while i < samples:
            while time > current_midi_time:
                if current_tick == np.shape(pianoroll)[0] - 1:
                    break
                current_tick = current_tick + 1
                current_midi_time = aux(current_tick)
            sampled_pianoroll[i] = pianoroll[current_tick]
            time = time + time_step
            i = i + 1

        return sampled_pianoroll

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
        # Length should include everything, including trailing silence
        return self._get_length_seconds(self._multitrack.get_maximal_length())

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
        for i in range(len(self._multitrack.tracks)):
            track = self._multitrack.tracks[i]
            track.pianoroll = track.pianoroll[:total_ticks]
            self._multitrack.tracks[i] = track

    def _get_length_seconds(self, ticks):
        seconds = 0
        for i in range(ticks):
            tempo = self._multitrack.tempo[i]
            resolution = self._multitrack.beat_resolution
            tick_time = 60/(tempo*resolution)
            seconds = seconds + tick_time
        return round(seconds, 1)

    def _get_ticks_from_seconds(self, duration):
        for i in range(self._multitrack.get_maximal_length()):
            seconds = self._get_length_seconds(i)
            if int(seconds) == int(duration):
                return i
