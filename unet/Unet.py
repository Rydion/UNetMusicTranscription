'''
author: Adrian Hintze @Rydion
'''

import tensorflow as tf

from .Encoder import Encoder
from .Decoder import Decoder

def sigmoid(inputs):
    return tf.nn.sigmoid(inputs)

def sigmoid_xentropy(logits, targets, weight):    
    ce = tf.nn.weighted_cross_entropy_with_logits(
        targets = targets,
        logits = logits,
        pos_weight = weight
    )
    return tf.reduce_mean(ce)

class UNetModel(object):
    def __init__(self, input, output, is_training, weight, kernel_size):
        self.input = input
        self.output = output
        self.is_training = is_training
        self._weight = weight
        self._optimizer = tf.train.AdamOptimizer(
            learning_rate = 0.0002,
            beta1 = 0.5
        )

        self.unet = UNet(
            self.input,
            self.is_training,
            kernel_size,
            'transcription-unet',
            reuse = False
        )

        self.prediction = self.unet.output
        self.cost = sigmoid_xentropy(self.prediction, self.output, self._weight)
      
        self.train_op = self._optimizer.minimize(self.cost)

class UNet(object):
    def __init__(self, input, is_training, kernel_size, name, reuse = False):
        with tf.variable_scope(name, reuse = reuse):
            self.encoder = Encoder(input, kernel_size, is_training, reuse)
            self.decoder = Decoder(self.encoder, kernel_size, is_training, reuse)
            self.output = sigmoid(self.decoder.output)
