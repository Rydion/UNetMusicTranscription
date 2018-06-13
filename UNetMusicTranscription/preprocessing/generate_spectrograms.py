'''
created on 2018-05-31
author: Adrian Hintze @Rydion
'''

import os
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile


DATA_PATH = './data/MIREX/'

# TODO: debug
#       normalize spectrogram

def plot_spectrogram(frequencies, times, spectrogram):
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def generate_signal_spectrogram(sample_rate, samples):
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    return frequencies, times, spectrogram

def read_wav(path):
    sample_rate, samples = wavfile.read(path)
    return sample_rate, samples

def generate_spectrograms(dir_path):
    for file in os.listdir(dir_path):
        _, file_extension = os.path.splitext(file)
        print(file_extension)
        if file_extension != '.wav':
            continue

        print('Generating spectrogram for %s .' % file)
        sample_rate, samples = read_wav(os.path.join(dir_path, file))
        frequencies, times, spectrogram = generate_signal_spectrogram(sample_rate, samples)

        # TODO: save the spectrogram

        plot_spectrogram(frequencies, times, spectrogram) # REMOVE
        break # REMOVE

def main():
    generate_spectrograms(DATA_PATH)

if __name__ == '__main__':
    main()
