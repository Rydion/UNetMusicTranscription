'''
created: 2018-06-08
edited: 2018-07-04
author: Adrian Hintze @Rydion
'''

import numpy as np

from PIL import Image

def to_rgb(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def crop_to_shape(data, shape):
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]

def combine_img_prediction(data, gt, pred):
    if np.isnan(pred).any():
        print('NaN prediction')
        pred = np.zeros(np.shape(pred))
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate(
        (
            to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)),
            to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)),
            to_rgb(pred[..., 1].reshape(-1, ny, 1))
        ),
        axis = 1
    )
    return img

def save_image(img, path):
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi = [300, 300], quality = 90)
