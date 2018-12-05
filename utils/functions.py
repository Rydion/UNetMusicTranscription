'''
created: 2018-06-14
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

def get_chunk_generator(matrix, chunk_length):
    for i in range(0, np.shape(matrix)[1], chunk_length):
        yield matrix[:, i:i + chunk_length]

def expand_array(arr, size, pixel_mult, dim):
    orig_size = np.shape(arr)
    if orig_size[dim] >= size[dim]:
        return arr

    result = np.zeros(size)
    for i in range(orig_size[dim]):
        for j in range(pixel_mult):
            if dim == 0:
                result[i*pixel_mult + j, :] = arr[i, :]
            else:
                result[:, i*pixel_mult + j] = arr[:, i]
    return result

def collapse_array(arr, size, pixel_div, dim):
    orig_size = np.shape(arr)
    if orig_size[dim] <= size[dim]:
        return arr

    result = np.zeros(size)
    i = 0
    for i in range(size[dim]):
        for j in range(pixel_div):
            if dim == 0:
                result[i, :] = result[i, :] + arr[i*pixel_div + j, :]
            else:
                result[:, i] = result[:, i] + arr[:, i*pixel_div + j]
    #result = (result//pixel_div).astype(arr.dtype)
    result = (result > pixel_div//2).astype(arr.dtype)
    return result

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
