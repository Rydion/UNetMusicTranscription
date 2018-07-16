'''
created: 2018-06-14
edited: 2018-07-16
author: Adrian Hintze @Rydion
'''

import numpy as np

def normalize_array(arr):
    arr_min = np.amin(arr)
    arr_max = np.amax(arr)
    norm_elem = lambda x: (x - arr_min)/(arr_max - arr_min)
    return np.asarray([norm_elem(x) for x in arr])

def grey_scale(rgb_img):
    return np.dot(rgb_img[..., 0:3], [0.299, 0.587, 0.114])

def binarize(grey_img, threshold):
    grey_img = grey_img.astype(np.uint8)
    return np.where(grey_img > threshold, 255, 0)
