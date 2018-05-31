'''
created on 2018-05-30
author: Adrian Hintze @Rydion
'''


import tensorflow as tf


def bias(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def weight(shape, stddev = 0.1):
    initial = tf.truncated_normal(shape, stddev = stddev)
    return tf.Variable(initial)

def weightDevonc(shape, stddev = 0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev = stddev))

def conv2d(x, W, keepProb):
    layer = tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')
    return tf.nn.dropout(conv_2d, keepProb)

def deconv2d(x, W, stride):
    xShape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, stride, stride, 1], padding = 'VALID')

def maxPool(x, n):
    return tf.nn.max_pool(x, ksize = [1, n, n, 1], strides = [1, n, n, 1], padding = 'VALID')

def cropAndConcat(x1, x2):
    x1Shape = tf.shape(x1)
    x2Shape = tf.shape(x2)
    offsets = [0, (x1Shape[1] - x2Shape[1])//2, (x1Shape[2] - x2Shape[2])//2, 0]
    size = [-1, x2Shape[1], x2Shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)
