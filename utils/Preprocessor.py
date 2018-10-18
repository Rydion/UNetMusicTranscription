'''
author: Adrian Hintze
'''

import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from utils.AudioReader import AudioReader
from utils.Stft import Stft
from utils.Cqt import Cqt
from utils.Midi import Midi
from utils.functions import grey_scale, binarize, get_chunk_generator, expand_array

#plt.rcParams['figure.figsize'] = [4, 18]
#plt.rcParams['figure.dpi'] = 32

# TODO:
#   support other extensions apart from wav
#   make the size of the output/input pair images a parameter

class Preprocessor:
    @staticmethod
    def calc_rounded_slice_length(num, dem):
        """
            Calculate the closest that would give a remainder of 0.
            This way we avoid weird results when slicing the spectrogram.
            We will end up with N or N - 1 slices (where N is the number of slices if num%dem == 0).
        """
        remainder = num%dem
        remainder = dem - remainder
        true_len = num + remainder
        return true_len//dem

    def __init__(self, src_dir, dst_dir, img_format, input_suffix, output_suffix, downsample_rate, samples_per_second):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.training_dst_dir = os.path.join(self.dst_dir, 'training')
        self.validation_dst_dir = os.path.join(self.dst_dir, 'validation')
        self.test_dst_dir = os.path.join(self.dst_dir, 'test')
        self.img_format = img_format
        self._input_suffix = input_suffix
        self._output_suffix = output_suffix
        self._downsample_rate = downsample_rate
        self._samples_per_second = samples_per_second

        self._fill_digits = 4
        self._cqt_stride = self._downsample_rate//self._samples_per_second
        self._stft_window_length = 1024
        self._stft_stride = self._stft_window_length//2
        self._gt_suffix = '.gt'

    def preprocess(self, gen_input = True, gen_output = True, transformation = 'stft', duration_multiplier = 1, color = False):
        self._delete_dst_dir()
        self._create_dst_dirs()

        for file in os.listdir(self.src_dir):
            file_name, file_extension = os.path.splitext(file)
            # Search for audio files
            if file_extension != '.wav':
                continue

            # Skip any input file that does not have a corresponding output
            if not os.path.isfile(os.path.join(self.src_dir, file_name + '.mid')):
                continue

            print('Generating input/output for %s.' % file_name)

            # Read original files
            midi = Midi.from_file(os.path.join(self.src_dir, file_name + '.mid'))
            duration = midi.get_length_seconds()

            sample_rate, samples = AudioReader.read_wav(
                os.path.join(self.src_dir, file),
                as_mono = True,
                downsample = True,
                downsample_rate = self._downsample_rate,
                duration = duration
            )

            if transformation == 'stft':
                spectrogram = Stft.from_audio(
                    sample_rate,
                    samples,
                    window_length = self._stft_window_length,
                    stride = self._stft_stride
                )
            elif transformation == 'cqt':
                spectrogram = Cqt.from_audio(
                    sample_rate,
                    samples,
                    stride = self._cqt_stride,
                    starting_note = 'C0',
                    num_octaves = 8,
                    bins_per_octave = 36
                )
            else:
                raise ValueError('Unknown transformation: ' + transformation + '.')

            spectrogram_img = spectrogram.values

            num_notes = 96
            midi_img = midi.get_pianoroll(self._samples_per_second, note_min = 12, num_notes = num_notes) # 8 octaves starting at C0
            samples = np.shape(midi_img)[1]
            pixels_per_note = np.shape(spectrogram_img)[0]//num_notes
            midi_img_expanded = expand_array(midi_img, (np.shape(spectrogram_img)[0], samples), pixels_per_note, 0)

            duration = int(duration)
            slice_length = np.shape(spectrogram_img)[1]//duration
            slice_length = slice_length*duration_multiplier

            if gen_input:
                #spectrogram.save(os.path.join(self.dst_dir, file_name + '.spectrogram.png'), duration, 84, color = color)
                chunks = get_chunk_generator(spectrogram_img, slice_length)
                self._save_sliced(chunks, file_name, file_suffix = self._input_suffix, binary = False)

            if gen_output:
                #midi.save(os.path.join(self.dst_dir, file_name + '.midi.png'), duration, plain = False)
                # UNET ground truth
                chunks = get_chunk_generator(midi_img_expanded, slice_length)
                self._save_sliced(chunks, file_name, file_suffix = self._output_suffix, binary = True)
                # Validation ground truth
                chunks = get_chunk_generator(midi_img, slice_length)
                self._save_sliced(chunks, file_name, file_suffix = self._gt_suffix, binary = True)

            # Split into train/test by class
            self._split()

            # Output aesthetics
            print()

    def _delete_dst_dir(self):
        if os.path.exists(self.dst_dir):
            shutil.rmtree(self.dst_dir)
            # Without the delay sometimes weird shit happens when deleting/creating the folder
            time.sleep(1)

    def _create_dst_dirs(self):
        os.makedirs(self.dst_dir)
        os.makedirs(self.training_dst_dir)
        os.makedirs(self.validation_dst_dir)
        os.makedirs(self.test_dst_dir)

    def _save_sliced(self, chunks, file_name, file_suffix = '', binary = False):
        slice_length = 0
        start = time.clock()
        for i, c in enumerate(chunks):
            # Don't save the last slice if it is smaller than the rest
            if np.shape(c)[1] < slice_length:
                break

            slice_length = np.shape(c)[1]

            final_length = 64
            c = expand_array(c, (np.shape(c)[0], final_length), final_length//np.shape(c)[1], 1)

            c = (c*255).astype(np.uint8)
            img = Image.fromarray(c, 'L').rotate(180)

            dst_file = os.path.join(self.dst_dir, file_name + '_' + str(i + 1).zfill(self._fill_digits) + file_suffix + self.img_format)
            img.save(dst_file)
        end = time.clock()
        print('Saved all slices in %.2f seconds.' % (end - start))

    def _split(self):
        files = []
        for file in os.listdir(self.dst_dir):
            if os.path.isdir(file):
                continue

            file_name, file_extension = os.path.splitext(file)
            if not file_name.endswith(self._input_suffix):
                continue

            input_file_name = file_name
            input_file = file
            output_file_name = os.path.splitext(file_name)[0] + self._output_suffix
            output_file = output_file_name + file_extension
            gt_file_name = os.path.splitext(file_name)[0] + self._gt_suffix
            gt_file = gt_file_name + file_extension
            if os.path.isfile(os.path.join(self.dst_dir, output_file)):
                files.append((os.path.join(self.dst_dir, input_file), os.path.join(self.dst_dir, output_file), os.path.join(self.dst_dir, gt_file)))

        dataset_size = len(files)
        training_dataset_size = int(0.8*dataset_size)
        validation_dataset_size = int(0.1*dataset_size)
        test_dataset_size = int(0.1*dataset_size)
        rounding_error = dataset_size - training_dataset_size - validation_dataset_size - test_dataset_size
        training_dataset_size = training_dataset_size + rounding_error

        np.random.shuffle(files)
        training_files, validation_files, test_files = files[:training_dataset_size], files[training_dataset_size:training_dataset_size + validation_dataset_size], files[training_dataset_size + validation_dataset_size:]

        for f in training_files:
            input_file, output_file, gt_file = f
            shutil.move(input_file, self.training_dst_dir)
            shutil.move(output_file, self.training_dst_dir)
            shutil.move(gt_file, self.training_dst_dir)
        for f in validation_files:
            input_file, output_file, gt_file = f
            shutil.move(input_file, self.validation_dst_dir)
            shutil.move(output_file, self.validation_dst_dir)
            shutil.move(gt_file, self.validation_dst_dir)
        for f in test_files:
            input_file, output_file, gt_file = f
            shutil.move(input_file, self.test_dst_dir)
            shutil.move(output_file, self.test_dst_dir)
            shutil.move(gt_file, self.test_dst_dir)
