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
    return tf.nn.leaky_relu(x, alpha = leak)
    '''
    with tf.variable_scope('lrelu'):
        f1 = 0.5*(1 + leak)
        f2 = 0.5*(1 - leak)
        return f1*x + f2*abs(x)
    '''

def batch_norm(inputs, is_training, reuse):
    return tf.layers.batch_normalization(
        inputs,
        momentum = 0.9,
        epsilon = 1e-5,
        training = is_training,
        reuse = reuse
    )

class Encoder(object):
    def __init__(self, input_tensor, kernel_size, is_training, reuse):
        net = input_tensor
        self.layers = []

        with tf.variable_scope('encoder'):
            n = 1
            filters = 9
            num_layers = 6
            while n <= num_layers:
                with tf.variable_scope('layer-{0}'.format(n)):
                    #print('Encoder. Layer {0}, filters {1}.'.format(n, filters))
                    if n > 1:
                        net = lrelu(net)
                    stride = (3, 2) if n == 1 else (2, 2)
                    #stride = (1, 2)
                    net = conv(net, filters = filters, kernel_size = kernel_size, stride = stride)
                    if n != num_layers:
                        net = batch_norm(net, is_training = is_training, reuse = reuse)
                        self.layers.append(net)
                filters = filters*2
                n = n + 1

            self.output = net
