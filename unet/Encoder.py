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

        with tf.variable_scope('encoder'):
            with tf.variable_scope('layer-1'):
                net = conv(net, filters = 16, kernel_size = 5, stride = (2, 2))
                self.l1 = net

            with tf.variable_scope('layer-2'):
                net = lrelu(net)
                net = conv(net, filters = 32, kernel_size = 5, stride = (2, 2))
                net = batch_norm(net, is_training = is_training, reuse = reuse)
                self.l2 = net

            with tf.variable_scope('layer-3'):
                net = lrelu(net)
                net = conv(net, filters=64, kernel_size=5, stride=(2, 2))
                net = batch_norm(net, is_training = is_training, reuse = reuse)
                self.l3 = net

            with tf.variable_scope('layer-4'):
                net = lrelu(net)
                net = conv(net, filters = 128, kernel_size = 5, stride = (2, 2))
                net = batch_norm(net, is_training=is_training, reuse = reuse)
                self.l4 = net

            with tf.variable_scope('layer-5'):
                net = lrelu(net)
                net = conv(net, filters = 256, kernel_size = 5, stride = (2, 2))
                net = batch_norm(net, is_training = is_training, reuse = reuse)
                self.l5 = net

            with tf.variable_scope('layer-6'):
                net = lrelu(net)
                net = conv(net, filters = 512, kernel_size = 5, stride = (2, 2))

            self.output = net
