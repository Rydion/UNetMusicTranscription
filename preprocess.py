'''
created: 2018-06-14
author: Adrian Hintze @Rydion
'''

import os

from utils.Preprocessor import Preprocessor

DURATION_MULTIPLIER = 4 # 1 for 1 second slices, 2 for 0.5 seconds, etc
TRANSFORMATION = 'cqt' # stft cqt
DATASET = 'MIREX' # Piano MIREX
DATA_SRC_PATH = os.path.join('./data/raw/', DATASET)
FULL_DATASET = '{0}.{1}.{2}'.format(DATASET, TRANSFORMATION, DURATION_MULTIPLIER)
DATASET_DST_PATH = os.path.join('./data/preprocessed/', FULL_DATASET)
 
def main():
    preprocessor = Preprocessor(DATA_SRC_PATH, DATASET_DST_PATH)
    preprocessor.preprocess(
        gen_input = True,
        gen_output = True,
        transformation = TRANSFORMATION,
        duration_multiplier = DURATION_MULTIPLIER
    )


if __name__ == '__main__':
    main()
