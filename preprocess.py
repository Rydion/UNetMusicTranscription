'''
author: Adrian Hintze @Rydion
'''

import os

from utils.Preprocessor import Preprocessor

COLOR = False
DURATION_MULTIPLIER = 4 # slices of 1/DURATION_MULTIPLIER seconds
TRANSFORMATION = 'cqt' # stft cqt
DATASET = 'smd' # mirex smd piano-score piano-correct amaps-akpnbcht maps-akpnbcht
DATA_SRC_PATH = os.path.join('./data/raw/', DATASET)
FULL_DATASET = '{0}.{1}.{2}'.format(DATASET, TRANSFORMATION, DURATION_MULTIPLIER)
DATASET_DST_PATH = os.path.join('./data/preprocessed/', FULL_DATASET)
 
def main():
    preprocessor = Preprocessor(DATA_SRC_PATH, DATASET_DST_PATH)
    preprocessor.preprocess(
        gen_input = True,
        gen_output = True,
        transformation = TRANSFORMATION,
        duration_multiplier = DURATION_MULTIPLIER,
        color = COLOR
    )


if __name__ == '__main__':
    main()
