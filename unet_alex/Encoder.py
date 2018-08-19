import tensorflow as tf

class Encoder(object):
    def __init__(self, input_tensor, is_training, reuse):
        net = input_tensor

        with tf.variable_scope('encoder'):
            with tf.variable_scope('layer-1'):
                net = conv(net, filters=16, kernel_size=5, stride=(2, 2))
                self.l1 = net

            with tf.variable_scope('layer-2'):
                net = lrelu(net)
                net = conv(net, filters=32, kernel_size=5, stride=(2, 2))
                net = batch_norm(net, is_training=is_training, reuse=reuse)
                self.l2 = net

            with tf.variable_scope('layer-3'):
                net = lrelu(net)
                net = conv(net, filters=64, kernel_size=5, stride=(2, 2))
                net = batch_norm(net, is_training=is_training, reuse=reuse)
                self.l3 = net

            with tf.variable_scope('layer-4'):
                net = lrelu(net)
                net = conv(net, filters=128, kernel_size=5, stride=(2, 2))
                net = batch_norm(net, is_training=is_training, reuse=reuse)
                self.l4 = net

            with tf.variable_scope('layer-5'):
                net = lrelu(net)
                net = conv(net, filters=256, kernel_size=5, stride=(2, 2))
                net = batch_norm(net, is_training=is_training, reuse=reuse)
                self.l5 = net

            with tf.variable_scope('layer-6'):
                net = lrelu(net)
                net = conv(net, filters=512, kernel_size=5, stride=(2, 2))

            self.output = net
