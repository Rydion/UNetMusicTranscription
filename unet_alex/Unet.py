import tensorflow as tf

from unet_alex.Encoder import Encoder
from unet_alex.Decoder import Decoder

def tanh(inputs):
    return tf.nn.tanh(inputs)

def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))

class UNetModel(object):
    def __init__(self, input, output, is_training):
        self.input = input
        self.is_training = is_training

        self.unet = UNet(
            input,
            is_training = is_training,
            name = 'transcription-unet',
            reuse = False
        )

        self.prediction = self.unet.output

        self.cost = l1_loss(self.prediction, output)

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
