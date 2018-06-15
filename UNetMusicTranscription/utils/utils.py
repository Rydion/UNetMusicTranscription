'''
created on 2018-06-14
author: Adrian Hintze @Rydion
'''

import numpy as np

def normalize_array(arr):
    arr_min = np.amin(arr)
    arr_max = np.amax(arr)
    norm_elem = lambda x: (x - arr_min)/(arr_max - arr_min)
    return np.asarray([norm_elem(x) for x in arr])
