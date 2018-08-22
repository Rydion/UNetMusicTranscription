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
    def __init__(self, input_tensor, encoder, is_training, reuse):
        net = input_tensor

        with tf.variable_scope('decoder'):
            with tf.variable_scope('layer-1'):
                net = relu(net)
                net = deconv(net, filters = 256, kernel_size = 5, stride = (2, 2))
                net = batch_norm(net, is_training = is_training, reuse = reuse)
                net = dropout(net, .5)

            with tf.variable_scope('layer-2'):
                net = relu(concat(net, encoder.l5))
                net = deconv(net, filters = 128, kernel_size = 5, stride = (2, 2))
                net = batch_norm(net, is_training = is_training, reuse = reuse)
                net = dropout(net, .5)

            with tf.variable_scope('layer-3'):
                net = relu(concat(net, encoder.l4))
                net = deconv(net, filters=64, kernel_size=5, stride = (2, 2))
                net = batch_norm(net, is_training = is_training, reuse = reuse)
                net = dropout(net, .5)

            with tf.variable_scope('layer-4'):
                net = relu(concat(net, encoder.l3))
                net = deconv(net, filters = 32, kernel_size = 5, stride = (2, 2))
                net = batch_norm(net, is_training = is_training, reuse = reuse)

            with tf.variable_scope('layer-5'):
                net = relu(concat(net, encoder.l2))
                net = deconv(net, filters = 16, kernel_size = 5, stride = (2, 2))
                net = batch_norm(net, is_training = is_training, reuse = reuse)

            with tf.variable_scope('layer-6'):
                net = relu(concat(net, encoder.l1))
                net = deconv(net, filters = 1, kernel_size = 5, stride = (2, 2))

            self.output = net
