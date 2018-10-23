'''
author: Adrian Hintze @Rydion
'''

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from Spectrogram import Spectrogram

class Cqt(Spectrogram):
    @classmethod
    def from_audio(cls, sample_rate, samples, stride = 512, starting_note = 'C0', num_octaves = 8, bins_per_octave = 36):
        values = librosa.core.cqt(
            samples,
            sr = sample_rate,
            fmin = librosa.note_to_hz(starting_note),
            n_bins = int(num_octaves*bins_per_octave),
            bins_per_octave = bins_per_octave,
            hop_length = stride
        )
        values = librosa.amplitude_to_db(np.abs(values), ref = np.max)
        values = (values - np.mean(values))/np.std(values)
        values = Cqt.normalize_values(values)
        return Cqt(values, sample_rate)

    def __init__(self, *args):
        super().__init__(*args)

    def _plot(self, mult_x, mult_y, color):
        figsize_x, figsize_y = plt.rcParams['figure.figsize']
        fig, ax = plt.subplots(1, figsize = (mult_x*figsize_x, figsize_y))
        librosa.display.specshow(
            self.values,
            sr = self.sample_rate,
            y_axis = 'cqt_note',
            x_axis = 'time',
            ax = ax,
            cmap = None if color else plt.cm.gray
        )
        return fig, ax
