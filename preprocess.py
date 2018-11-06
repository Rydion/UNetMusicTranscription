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
    input_suffix,
    output_suffix,
    gt_suffix,
    transformation,
    downsample_rate,
    samples_per_second,
    duration_multiplier
):
    preprocessor = Preprocessor(
        data_src_path,
        dataset_dst_path,
        img_format,
        input_suffix,
        output_suffix,
        gt_suffix,
        downsample_rate,
        samples_per_second
    )
    preprocessor.preprocess(
        gen_input = True,
        gen_output = True,
        transformation = transformation,
        duration_multiplier = duration_multiplier,
    )

if __name__ == '__main__':
    print('Starting preprocessor...')

    conf = configparser.ConfigParser()
    conf.read('conf.ini')

    # global conf
    global_conf = conf['global']
    DATASET = global_conf['dataset']
    DURATION_MULTIPLIER = int(global_conf['multiplier'])
    TRANSFORMATION = global_conf['transformation']
    IMG_FORMAT = global_conf['format']
    INPUT_SUFFIX = global_conf['input_suffix']
    OUTPUT_SUFFIX = global_conf['output_suffix']
    GT_SUFFIX = global_conf['gt_suffix']

    # preprocessing conf
    process_conf = conf['processing']
    DOWNSAMPLE_RATE = int(process_conf['downsample'])
    SAMPLES_PER_SECOND = int(process_conf['samples_per_second'])

    # paths
    DATA_SRC_PATH = os.path.join('./data/raw/', DATASET)

    FULL_DATASET = '{0}.{1}.dr-{2}.sps-{3}.dm-{4}'.format(DATASET, TRANSFORMATION, DOWNSAMPLE_RATE, SAMPLES_PER_SECOND, DURATION_MULTIPLIER)
    DATASET_DST_PATH = os.path.join('./data/preprocessed/', FULL_DATASET)

    print(DATASET_DST_PATH)

    main(
        DATA_SRC_PATH,
        DATASET_DST_PATH,
        IMG_FORMAT,
        INPUT_SUFFIX,
        OUTPUT_SUFFIX,
        GT_SUFFIX,
        TRANSFORMATION,
        DOWNSAMPLE_RATE,
        SAMPLES_PER_SECOND,
        DURATION_MULTIPLIER
    )
