'''
created on 2018-05-31
author: Adrian Hintze @Rydion
'''

import tensorflow as tf
from unet.Unet import Unet

def init():
    tf.reset_default_graph()
    unet = Unet()

def main():
    print('Python sucks dick')
    init()


if __name__ == "__main__":
    main()
