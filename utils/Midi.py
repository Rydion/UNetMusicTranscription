'''
created: 2018-06-15
author: Adrian Hintze @Rydion
'''

import matplotlib.pyplot as plt

import pypianoroll as pyano

class Midi:
    @staticmethod
    def from_file(file_path):
        multitrack = pyano.parse(file_path)
        return Midi(multitrack)

    def __init__(self, multitrack):
        self._multitrack = multitrack

    def plot(self):
        fig, ax = pyano.plot(
            self.midi,
            preset = 'plain',
            ytick = 'pitch'
        )
        plt.show(fig)

    @property
    def midi(self):
        return self._multitrack
