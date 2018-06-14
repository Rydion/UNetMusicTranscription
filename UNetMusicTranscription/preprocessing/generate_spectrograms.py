'''
created on 2018-06-14
author: Adrian Hintze @Rydion
'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile
from utils import utils

DATA_PATH = './data/MIREX/'
DOWNSAMPLE_RATE = 8192 #Hz
SFFT_WINDOW_LENGTH = 1024
SFFT_STRIDE = 768 #SFFT_WINDOW_LENGTH//2

# TODO: debug
#       normalize spectrogram
#       use logging instead of prints

def plot_spectrogram(frequencies, times, spectrogram):
    db_spectrogram = 10*np.log10(spectrogram)
    plt.pcolormesh(times, frequencies, db_spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def downsample_signal(X, original_sample_rate, new_sample_rate):
    secs = len(X)/original_sample_rate
    num_samples = secs*new_sample_rate
    num_samples = int(num_samples)
    return signal.resample(X, num_samples)

def normalize_spectrogram(spectrogram):
    return utils.normalize_array(spectrogram)

def generate_signal_spectrogram(sample_rate, samples):
    print('Signal with a sample rate of %i' % sample_rate)

    if sample_rate < DOWNSAMPLE_RATE:
        raise ValueError('The sample rate of the audio should be at least %i. It is %i.' % DOWNSAMPLE_RATE, sample_rate)
    
    if sample_rate > DOWNSAMPLE_RATE:
        print('Resampling to %i' % DOWNSAMPLE_RATE)
        start = time.clock()
        sample_rate = DOWNSAMPLE_RATE
        samples = downsample_signal(samples, sample_rate, DOWNSAMPLE_RATE)
        end = time.clock()
        print('Downsampled in %.2f seconds.' % (end - start))
    
    return signal.spectrogram(
        samples,
        fs = sample_rate,
        nperseg = SFFT_WINDOW_LENGTH,
        noverlap = SFFT_STRIDE
    )

def read_wav(path):
    sample_rate, samples = wavfile.read(path)
    # TODO if sound is stereo transform to mono
    # either by keeping only one channel (trivial) or combining them (not trivial)
    return sample_rate, samples

def generate_spectrograms(dir_path):
    for file in os.listdir(dir_path):
        _, file_extension = os.path.splitext(file)
        if file_extension != '.wav':
            continue

        print('Generating spectrogram for %s .' % file)
        sample_rate, samples = read_wav(os.path.join(dir_path, file))
        frequencies, times, spectrogram = generate_signal_spectrogram(sample_rate, samples)
        spectrogram = normalize_spectrogram(spectrogram)
        # TODO: save the spectrogram

        plot_spectrogram(frequencies, times, spectrogram) # REMOVE
        break # REMOVE

def main():
    generate_spectrograms(DATA_PATH)

if __name__ == '__main__':
    main()
