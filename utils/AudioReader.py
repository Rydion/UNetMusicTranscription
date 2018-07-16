'''
created: 2018-06-15
edited: 2018-07-16
author: Adrian Hintze @Rydion
'''

import time
import numpy as np

from scipy import signal
from scipy.io import wavfile

# TODO:
#   switch for verbosity

class AudioReader:
    @staticmethod
    def calc_signal_length_seconds(signal, sample_rate):
        return len(signal)//sample_rate

    @staticmethod
    def stereo_to_mono(samples):
        # If it is already a mono sound
        if np.ndim(samples) == 1:
            return samples

        samples = samples.astype(float)
        # TODO check mono compatibility?
        # Alternatively we could just pick one of the channels
        return (samples[:, 0] + samples[:, 1])//2

    @staticmethod
    def downsample_signal(X, original_sample_rate, new_sample_rate):
        print('Signal with a sample rate of %i.' % original_sample_rate)
    
        if original_sample_rate < new_sample_rate:
            raise ValueError('The sample rate of the signal (%i) should be higher than the new sample rate (%i).' % original_sample_rate, new_sample_rate)

        if original_sample_rate == new_sample_rate:
            return original_sample_rate, X

        print('Downsampling to %i.' % new_sample_rate)
        start = time.clock()
        seconds = AudioReader.calc_signal_length_seconds(X, original_sample_rate)
        num_samples = seconds*new_sample_rate
        Y = signal.resample(X, num_samples)
        end = time.clock()
        print('Downsampled in %.2f seconds.' % (end - start))

        return new_sample_rate, Y

    def read_wav(self, file_path, as_mono = False, downsample = False, downsample_rate = 8192):
        print('Reading %s.' % file_path)
        sample_rate, samples = wavfile.read(file_path)
        if as_mono:
            samples = AudioReader.stereo_to_mono(samples)
        if downsample:
            sample_rate, samples = AudioReader.downsample_signal(samples, sample_rate, downsample_rate)
        return sample_rate, samples
