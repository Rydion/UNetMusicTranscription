import tensorflow as tf

from unet_alex.Encoder import Encoder
from unet_alex.Decoder import Decoder

def tanh(inputs):
    return tf.nn.tanh(inputs)

def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))

def l2_loss(x, y):
    return tf.nn.l2_loss(tf.abs(x - y))

def sigmoid_xentropy(x, y):
    ce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = y,
        logits = x
    )
    return tf.reduce_sum(tf.abs(ce))

def softmax_xentropy(x, y):
    ce = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits = tf.reshape(x, [-1, 2]),
        labels = tf.reshape(y, [-1, 2])
    )
    return tf.reduce_sum(tf.abs(ce))

def normalize(x):
    min = tf.reduce_min(x)
    max = tf.reduce_max(x)
    return tf.div(
        tf.subtract(x, min),
        tf.subtract(max, min)
    )

class UNetModel(object):
    def __init__(self, input, output, is_training):
        self.input = input
        self.output = output
        self.is_training = is_training

        self.unet = UNet(
            self.input,
            is_training = self.is_training,
            name = 'transcription-unet',
            reuse = False
        )

        self.prediction = self.unet.output

        #self.cost = l1_loss(self.prediction, output)
        #self.cost = l2_loss(self.prediction, output)
        self.cost = sigmoid_xentropy(self.prediction, self.output)
        #self.cost = softmax_xentropy(self.prediction, output)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = 0.0002,
            beta1 = 0.5
        )
        self.train_op = self.optimizer.minimize(self.cost)

class UNet(object):
    def __init__(self, input, is_training, name, reuse = False):
        with tf.variable_scope(name, reuse = reuse):
            self.encoder = Encoder(input, is_training, reuse)
            self.decoder = Decoder(self.encoder.output, self.encoder, is_training, reuse)
            self.output = tanh(self.decoder.output)/2 + 0.5
