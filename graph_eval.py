'''
author: Adrian Hintze
'''

import os
import warnings
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Suppress unnecessary warnings from libraries
warnings.filterwarnings('ignore')

SRC_DIRS = [
    './graphs/smd/',
    './graphs/maps/'
]

if __name__ == '__main__':
    for i, dir in enumerate(SRC_DIRS):
        eval = None 
        with open(os.path.join(dir, 'best_model.pkl'), 'rb') as f:
            results = pickle.load(f)
            eval = results['eval']

        print(dir)
        for k in eval:
            print(k)
            for l in eval[k]:
                print('  {0}'.format(l))
                print('  Precision {0}'.format(eval[k][l]['precision']))
                print('  Recall {0}'.format(eval[k][l]['recall']))
                print('  F1 {0}'.format(eval[k][l]['fscore']))
