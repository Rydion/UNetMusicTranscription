'''
author: Adrian Hintze @Rydion
'''

import tensorflow as tf

def deconv(inputs, filters, kernel_size, stride):
    return tf.layers.conv2d_transpose(
        inputs,
        filters = filters,
        kernel_size = kernel_size,
        kernel_initializer = tf.random_normal_initializer(stddev = 0.02),
        strides = stride,
        padding = 'SAME'
    )

def relu(inputs):
    return tf.nn.relu(inputs)

def concat(x, y):
    return tf.concat([x, y], axis = 3)

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

def dropout(inputs, rate):
    return tf.nn.dropout(inputs, keep_prob = 1 - rate)

class Decoder(object):
    def __init__(self, encoder, kernel_size, is_training, reuse):
        net = encoder.output

        with tf.variable_scope('decoder'):
            n = 1
            filters = 144
            num_layers = 6
            while n <= num_layers:
                stride = (3, 2) if (n == num_layers) else (2, 2)
                with tf.variable_scope('layer-{0}'.format(n)):
                    print(['layer-{0}'.format(n), filters, stride])
                    net = relu(net) if n == 1 else relu(concat(net, encoder.layers[num_layers - n]))
                    net = deconv(net, filters = filters, kernel_size = kernel_size, stride = stride)
                    if n < num_layers:
                        net = batch_norm(net, is_training = is_training, reuse = reuse)
                    if n <= 3:
                        net = dropout(net, 0.5)
                filters = 1 if (n == num_layers - 1) else filters//2
                n = n + 1

            self.output = net
