'''
created on 2018-06-14
author: Adrian Hintze @Rydion
'''

from __future__ import division

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile
from utils import utils

DATA_SRC_PATH = './data/raw/MIREX/'
DATA_DEST_PATH = './data/preprocessed/MIREX/'
DOWNSAMPLE_RATE = 8192 #Hz
SFFT_WINDOW_LENGTH = 1024
SFFT_STRIDE = 768 #SFFT_WINDOW_LENGTH//2

# TODO: debug
#       use logging instead of prints

'''
def save_spectrogram(sample_rate, samples, file_name):
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
    ax.axis('off')
    _, _, _, _ = ax.specgram(x = samples, Fs = sample_rate, noverlap = SFFT_STRIDE, NFFT = SFFT_WINDOW_LENGTH)
    ax.axis('off')
    fig.savefig(os.path.join(DATA_DEST_PATH, file_name + '.png'), dpi = 300, frameon = 'false')
'''

def save_spectrogram(frequencies, times, spectrogram, file_name):
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
    ax.axis('off')
    ax.pcolormesh(times, frequencies, spectrogram)
    ax.axis('off')
    fig.savefig(os.path.join(DATA_DEST_PATH, file_name + '.png'), dpi = 300, frameon = 'false')

def plot_spectrogram(frequencies, times, spectrogram):
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def normalize_spectrogram(spectrogram):
    return utils.normalize_array(spectrogram)

def downsample_signal(X, original_sample_rate, new_sample_rate):
    print('Signal with a sample rate of %i.' % original_sample_rate)
    
    if original_sample_rate < new_sample_rate:
        raise ValueError('The sample rate of the signal (%i) should be higher than the new sample rate (%i).' % original_sample_rate, new_sample_rate)

    if original_sample_rate == new_sample_rate:
        return original_sample_rate, X

    print('Downsampling to %i.' % new_sample_rate)
    start = time.clock()
    seconds = len(X)/original_sample_rate
    num_samples = seconds*new_sample_rate
    num_samples = int(num_samples)
    Y = signal.resample(X, num_samples)
    end = time.clock()
    print('Downsampled in %.2f seconds.' % (end - start))

    return new_sample_rate, Y

def generate_signal_spectrogram(sample_rate, samples):
    frequencies, times, spectrogram = signal.spectrogram(
        samples,
        fs = sample_rate,
        nperseg = SFFT_WINDOW_LENGTH,
        noverlap = SFFT_STRIDE
    )
    spectrogram = 20*np.log10(spectrogram)
    spectrogram = normalize_spectrogram(spectrogram)
    return frequencies, times, spectrogram

def wav_to_mono(samples):
    # If it is already a mono sound
    if samples.ndim == 1:
        return samples

    samples = samples.astype(float)
    # TODO should we check mono compatibility?
    # Alternatively we could just pick one of the channels
    return (samples[:, 0] + samples[:, 1])/2

def read_wav(file_path, downsample = False):
    sample_rate, samples = wavfile.read(file_path)
    samples = wav_to_mono(samples)
    if downsample:
        return downsample_signal(samples, sample_rate, DOWNSAMPLE_RATE)

    return sample_rate, samples

def generate_spectrograms(dir_path, plot = False):
    for file in os.listdir(dir_path):
        file_name, file_extension = os.path.splitext(file)
        if file_extension != '.wav':
            continue

        print('Generating spectrogram for %s.' % file)
        file_path = os.path.join(dir_path, file)
        sample_rate, samples = read_wav(file_path, downsample = True)
        frequencies, times, spectrogram = generate_signal_spectrogram(sample_rate, samples)
        if plot:
            plot_spectrogram(frequencies, times, spectrogram)
        save_spectrogram(frequencies, times, spectrogram, file_name)

def main():
    generate_spectrograms(DATA_SRC_PATH)

if __name__ == '__main__':
    main()
