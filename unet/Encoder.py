'''
author: Adrian Hintze @Rydion
'''

import tensorflow as tf

def conv(inputs, filters, kernel_size, stride):
    return tf.layers.conv2d(
        inputs,
        filters = filters,
        kernel_size = kernel_size,
        kernel_initializer = tf.random_normal_initializer(stddev = 0.02),
        strides = stride,
        padding = 'SAME'
    )

def lrelu(x, leak = 0.2):
    with tf.variable_scope('lrelu'):
        f1 = 0.5*(1 + leak)
        f2 = 0.5*(1 - leak)
        return f1*x + f2*abs(x)

def batch_norm(inputs, is_training, reuse):
    return tf.contrib.layers.batch_norm(
        inputs,
        decay = 0.9,
        updates_collections = None,
        epsilon = 1e-5,
        scale = True,
        is_training = is_training,
        reuse = reuse
    )

class Encoder(object):
    def __init__(self, input_tensor, is_training, reuse):
        net = input_tensor

        kernel_size = (3, 7)
        stride = (1, 2)
        with tf.variable_scope('encoder'):
            with tf.variable_scope('layer-1'):
                net = conv(net, filters = 9, kernel_size = kernel_size, stride = stride)
                self.l1 = net

            with tf.variable_scope('layer-2'):
                net = lrelu(net)
                net = conv(net, filters = 18, kernel_size = kernel_size, stride = stride)
                net = batch_norm(net, is_training = is_training, reuse = reuse)
                self.l2 = net

            with tf.variable_scope('layer-3'):
                net = lrelu(net)
                net = conv(net, filters = 36, kernel_size = kernel_size, stride = stride)
                net = batch_norm(net, is_training = is_training, reuse = reuse)
                self.l3 = net

            with tf.variable_scope('layer-4'):
                net = lrelu(net)
                net = conv(net, filters = 72, kernel_size = kernel_size, stride = stride)
                net = batch_norm(net, is_training = is_training, reuse = reuse)
                self.l4 = net

            with tf.variable_scope('layer-5'):
                net = lrelu(net)
                net = conv(net, filters = 144, kernel_size = kernel_size, stride = stride)
                net = batch_norm(net, is_training = is_training, reuse = reuse)
                self.l5 = net

            with tf.variable_scope('layer-6'):
                net = lrelu(net)
                net = conv(net, filters = 288, kernel_size = kernel_size, stride = stride)

            self.output = net
