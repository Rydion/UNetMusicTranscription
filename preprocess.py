'''
author: Adrian Hintze
'''

import os
import configparser

from utils.Preprocessor import Preprocessor

def main(
    data_src_path,
    dataset_dst_path,
    img_format,
    transformation,
    downsample_rate,
    duration_multiplier,
    color
):
    preprocessor = Preprocessor(data_src_path, dataset_dst_path, img_format, downsample_rate)
    preprocessor.preprocess(
        gen_input = True,
        gen_output = True,
        transformation = transformation,
        duration_multiplier = duration_multiplier,
        color = color
    )

if __name__ == '__main__':
    print('Starting preprocessor...')

    conf = configparser.ConfigParser()
    conf.read('conf.ini')

    # global conf
    global_conf = conf['global']
    COLOR = global_conf.getboolean('color')
    DATASET = global_conf['dataset']
    DURATION_MULTIPLIER = int(global_conf['multiplier'])
    TRANSFORMATION = global_conf['transformation']
    IMG_FORMAT = global_conf['format']

    # preprocessing conf
    process_conf = conf['processing']
    DOWNSAMPLE_RATE = int(process_conf['downsample'])

    # paths
    DATA_SRC_PATH = os.path.join('./data/raw/', DATASET)

    FULL_DATASET = '{0}.{1}.{2}.{3}'.format(DATASET, TRANSFORMATION, DOWNSAMPLE_RATE, DURATION_MULTIPLIER)
    DATASET_DST_PATH = os.path.join('./data/preprocessed/', FULL_DATASET)

    print(DATASET_DST_PATH)

    main(
        DATA_SRC_PATH,
        DATASET_DST_PATH,
        IMG_FORMAT,
        TRANSFORMATION,
        DOWNSAMPLE_RATE,
        DURATION_MULTIPLIER,
        COLOR
    )
