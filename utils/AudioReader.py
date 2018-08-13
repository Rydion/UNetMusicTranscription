'''
created: 2018-06-15
author: Adrian Hintze @Rydion
'''

import librosa

class AudioReader:
    @staticmethod
    def calc_signal_length_seconds(signal, sample_rate):
        return len(signal)//sample_rate

    @staticmethod
    def read_wav(file_path, as_mono = False, downsample = False, downsample_rate = 16000):
        print('Reading %s.' % file_path)
        samples, sample_rate = librosa.load(
            file_path,
            mono = as_mono,
            sr = downsample_rate if downsample else None
        )
        return sample_rate, samples
