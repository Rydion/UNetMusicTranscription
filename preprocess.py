'''
created: 2018-06-14
author: Adrian Hintze @Rydion
'''

import os

from utils.Preprocessor import Preprocessor

TRANSFORMATION = 'cqt' # stft cqt
DATASET = 'MIREX' # Piano MIREX
DATA_SRC_PATH = os.path.join('./data/raw/', DATASET) 
DATA_DST_PATH = os.path.join('./data/preprocessed/', DATASET + '.' + TRANSFORMATION)

def main():
    preprocessor = Preprocessor()
    preprocessor.preprocess(
        DATA_SRC_PATH,
        DATA_DST_PATH,
        gen_input = True,
        gen_output = True,
        transformation = TRANSFORMATION
    )


if __name__ == '__main__':
    main()
