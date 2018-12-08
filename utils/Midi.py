'''
author: Adrian Hintze
'''

import numpy as np
import matplotlib.pyplot as plt
import pretty_midi

# https://github.com/craffel/pretty-midi/issues/112
pretty_midi.pretty_midi.MAX_TICK = 1e10 # Pretty MIDI does some stuff internally that doesn't allow to load MAPS

class Midi:
    @staticmethod
    def from_file(file_path):
        midi = pretty_midi.PrettyMIDI(file_path)
        return Midi(midi)

    def __init__(self, pm):
        self._pm = pm

    def get_length_seconds(self):
        return np.shape(self.get_pianoroll(128))[1]/128

    def get_pianoroll(self, samples_per_second, note_min = 0, num_notes = 128):
        pr = self._pm.get_piano_roll(fs = samples_per_second)
        pr = np.where(pr > 0, 1, 0)

        x = np.shape(pr)[1]%samples_per_second
        pr = pr[:, :-x]
        pr = pr[note_min:note_min + num_notes, :]

        return pr
