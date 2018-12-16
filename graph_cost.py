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
    './graphs/smd.cqt.dr-16384.sps-128.dm-1.ne-300.bs-128.grid-search/',
    './graphs/smd.cqt.dr-16384.sps-128.dm-2.ne-300.bs-128.grid-search/',
    './graphs/smd.cqt.dr-16384.sps-128.dm-3.ne-300.bs-128.grid-search/',
    './graphs/smd.cqt.dr-16384.sps-128.dm-4.ne-300.bs-128.grid-search/',
    './graphs/smd.cqt.dr-16384.sps-128.dm-5.ne-300.bs-128.grid-search/',
    './graphs/maps-enstdkcl.cqt.dr-16384.sps-128.dm-1.ne-300.bs-128.grid-search/'
]

if __name__ == '__main__':
    for i, dir in enumerate(SRC_DIRS):
        i = i + 1
        xs = []
        ys = []
        training_costs = []
        validation_costs = []
        test_costs = []
        fscores = []

        training_min = [np.inf, None]
        validation_min = [np.inf, None]
        test_min = [np.inf, None]
        training_max = [0, None]

        for file in os.listdir(dir):
            _, file_extension = os.path.splitext(file)
            if file_extension == '.png':
                continue

            results = None
            with open(os.path.join(dir, file), 'rb') as f:
                results = pickle.load(f)
                xs.append(results['params']['ks'][0])
                ys.append(results['params']['w'])
                training_costs.append(results['training_cost'])
                validation_costs.append(results['validation_cost'])
                test_costs.append(results['test_cost'])
                fscores.append(results['fscore'])

                if results['training_cost'] < training_min[0]:
                    training_min[0] = results['training_cost']
                    training_min[1] = file
                if results['training_cost'] > training_max[0]:
                    training_max[0] = results['training_cost']
                    training_max[1] = file

                if results['validation_cost'] < validation_min[0]:
                    validation_min[0] = results['validation_cost']
                    validation_min[1] = file

                if results['test_cost'] < test_min[0]:
                    test_min[0] = results['test_cost']
                    test_min[1] = file

        print(i)
        print('Training min {0} {1}'.format(training_min[0], training_min[1]))
        print('Training max {0} {1}'.format(training_max[0], training_max[1]))
        print('Validation min {0} {1}'.format(validation_min[0], validation_min[1]))
        print('Test min {0} {1}'.format(test_min[0], test_min[1]))
        
        xs = np.array(xs)
        ys = np.array(ys)
        training_costs = np.array(training_costs)
        validation_costs = np.array(validation_costs)
        test_costs = np.array(test_costs)
        fscores = np.array(fscores)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        ax.scatter(xs, ys, test_costs, marker = '^', color = 'r', label = 'Test Cost')
        ax.scatter(xs, ys, validation_costs, marker = '>', color = 'y', label = 'Validation Cost')
        ax.scatter(xs, ys, training_costs, marker = 'v', color = 'g', label = 'Training Cost')
        #ax.scatter(xs, ys, fscores, marker = 'o', color = 'b')

        ax.set_xlabel('Kernel Size')
        ax.set_ylabel('Weight')
        ax.set_zlabel('Cost')

        ax.legend(loc = 'upper left')

        #plt.show(fig)
        fig.savefig(os.path.join(dir, 'costs.{0}.png'.format(i)))
