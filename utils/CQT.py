'''
author: Adrian Hintze @Rydion
'''

import librosa
import numpy as np
import matplotlib.pyplot as plt

from utils.Spectrogram import Spectrogram

class CQT(Spectrogram):
    @classmethod
    def from_audio(cls, sample_rate, samples, stride = 512):
        values = librosa.core.cqt(
            samples,
            sr = sample_rate,
            #n_bins = 7*12, # total num of bins
            #bins_per_octave = 12,
            #hop_length = stride
        )
        values = librosa.amplitude_to_db(np.abs(values), ref = np.max)
        #values = CQT.normalize_values(values)
        return CQT(values, sample_rate)

    def __init__(self, *args):
        super().__init__(*args)

    def _plot(self, x, y, color = True):
        fig, ax = plt.subplots(1, figsize = (x*4, 16), dpi = 32)
        librosa.display.specshow(
            self.values,
            sr = self.sample_rate,
            y_axis = 'cqt_note',
            x_axis = 'time',
            ax = ax,
            cmap = None if color else plt.cm.gray
        )
        return fig, ax
