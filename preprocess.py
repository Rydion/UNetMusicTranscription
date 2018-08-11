'''
created: 2018-06-14
author: Adrian Hintze @Rydion
'''

from utils.Preprocessor import Preprocessor

DATA_SRC_PATH = './data/raw/Piano/'
DATA_DST_PATH = './data/preprocessed/Piano/'

def main():
    preprocessor = Preprocessor()
    preprocessor.preprocess(
        DATA_SRC_PATH,
        DATA_DST_PATH,
        gen_input = True,
        gen_output = True,
        transformation = 'spectrogram'
    )


if __name__ == '__main__':
    main()
