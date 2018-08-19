import tensorflow as tf

class Decoder(object):
    def __init__(self, input_tensor, encoder, is_training, reuse):
        net = input_tensor

        with tf.variable_scope('decoder'):
            with tf.variable_scope('layer-1'):
                net = relu(net)
                net = deconv(net, filters=256, kernel_size=5, stride=(2, 2))
                net = batch_norm(net, is_training=is_training, reuse=reuse)
                net = dropout(net, .5)

            with tf.variable_scope('layer-2'):
                net = relu(concat(net, encoder.l5))
                net = deconv(net, filters=128, kernel_size=5, stride=(2, 2))
                net = batch_norm(net, is_training=is_training, reuse=reuse)
                net = dropout(net, .5)

            with tf.variable_scope('layer-3'):
                net = relu(concat(net, encoder.l4))
                net = deconv(net, filters=64, kernel_size=5, stride=(2, 2))
                net = batch_norm(net, is_training=is_training, reuse=reuse)
                net = dropout(net, .5)

            with tf.variable_scope('layer-4'):
                net = relu(concat(net, encoder.l3))
                net = deconv(net, filters=32, kernel_size=5, stride=(2, 2))
                net = batch_norm(net, is_training=is_training, reuse=reuse)

            with tf.variable_scope('layer-5'):
                net = relu(concat(net, encoder.l2))
                net = deconv(net, filters=16, kernel_size=5, stride=(2, 2))
                net = batch_norm(net, is_training=is_training, reuse=reuse)

            with tf.variable_scope('layer-6'):
                net = relu(concat(net, encoder.l1))
                net = deconv(net, filters=1, kernel_size=5, stride=(2, 2))

            self.output = net
