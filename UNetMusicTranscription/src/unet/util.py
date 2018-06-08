'''
Created on 2018-06-08
author: Adrian Hintze @Rydion
'''

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from PIL import Image


def toRgb(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def cropToShape(data, shape):
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]

def combineImgPrediction(data, gt, pred):
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate(
        toRgb(cropToShape(data, pred.shape).reshape(-1, ny, ch), 
        toRgb(cropToShape(gt[..., 1], pred.shape).reshape(-1, ny, 1)), 
        toRgb(pred[..., 1].reshape(-1, ny, 1))),
        axis = 1
    )
    return img

def saveImage(img, path):
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi = [300, 300], quality = 90)
