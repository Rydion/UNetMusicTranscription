'''
author: Adrian Hintze @Rydion
'''

import os
import configparser

from utils.Preprocessor import Preprocessor
 
def main(
    data_src_path,
    dataset_dst_path,
    img_format,
    transformation,
    duration_multiplier,
    color
):
    preprocessor = Preprocessor(data_src_path, dataset_dst_path, img_format)
    preprocessor.preprocess(
        gen_input = True,
        gen_output = True,
        transformation = transformation,
        duration_multiplier = duration_multiplier,
        color = color
    )


if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read('conf.ini')

    global_conf = conf['global']
    COLOR = global_conf.getboolean('color')
    DATASET = global_conf['dataset']
    DURATION_MULTIPLIER = int(global_conf['multiplier'])
    TRANSFORMATION = global_conf['transformation']
    IMG_FORMAT = global_conf['format']

    DATA_SRC_PATH = os.path.join('./data/raw/', DATASET)

    FULL_DATASET = '{0}.{1}.{2}'.format(DATASET, TRANSFORMATION, DURATION_MULTIPLIER)
    DATASET_DST_PATH = os.path.join('./data/preprocessed/', FULL_DATASET)

    main(
        DATA_SRC_PATH,
        DATASET_DST_PATH,
        IMG_FORMAT,
        TRANSFORMATION,
        DURATION_MULTIPLIER,
        COLOR
    )
