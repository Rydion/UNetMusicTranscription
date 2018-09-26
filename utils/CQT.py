'''
author: Adrian Hintze @Rydion
'''

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from utils.Spectrogram import Spectrogram

class Cqt(Spectrogram):
    @classmethod
    def from_audio(cls, sample_rate, samples, stride):
        values = librosa.core.cqt(
            samples,
            sr = sample_rate,
            fmin = librosa.note_to_hz('C1'),
            n_bins = 252, # total num of bins
            bins_per_octave = 252//7,
            hop_length = stride
        )
        values = librosa.amplitude_to_db(np.abs(values), ref = np.max)
        #values = CQT.normalize_values(values)
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
