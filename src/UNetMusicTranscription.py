'''
created on 2018-05-31
author: Adrian Hintze @Rydion
'''

from __future__ import division, print_function

import shutil
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from unet.Unet import Unet
from src.unet.Trainer import Trainer
from tf_unet import image_gen


def init():
    tf.reset_default_graph()

def main():
    init()

if __name__ == '__main__':
    main()
