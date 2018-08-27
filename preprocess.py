'''
created: 2018-06-14
author: Adrian Hintze @Rydion
'''

import os

from utils.Preprocessor import Preprocessor

DURATION_MULTIPLIER = 2 # 1 for 1 second slices, 2 for 0.5 seconds, etc
TRANSFORMATION = 'cqt' # stft cqt
DATASET = 'MIREX' # Piano MIREX
DATA_SRC_PATH = os.path.join('./data/raw/', DATASET) 
DATA_DST_PATH = os.path.join('./data/preprocessed/', '{0}.{1}.{2}'.format(DATASET, TRANSFORMATION, DURATION_MULTIPLIER))
 
def main():
    preprocessor = Preprocessor()
    preprocessor.preprocess(
        DATA_SRC_PATH,
        DATA_DST_PATH,
        gen_input = True,
        gen_output = True,
        transformation = TRANSFORMATION,
        duration_multiplier = DURATION_MULTIPLIER
    )


if __name__ == '__main__':
    main()
