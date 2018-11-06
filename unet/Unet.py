'''
author: Adrian Hintze @Rydion
'''

import tensorflow as tf

from .Encoder import Encoder
from .Decoder import Decoder

def sigmoid(inputs):
    return tf.nn.sigmoid(inputs)

def sigmoid_xentropy(logits, targets, weight):
    #weights = tf.equal(y, 1.0)

    #note_pixels = tf.count_nonzero(weights, dtype = tf.int32)
    #total_pixels = tf.size(weights, out_type = tf.int32)
    #ratio = note_pixels/(total_pixels + note_pixels)

    #floatWeights = tf.cast(weights, tf.float32)
    #trueCase = tf.ones_like(floatWeights)*(total_pixels - note_pixels)
    #falseCase = tf.ones_like(floatWeights)*note_pixels
    #trueCase = tf.ones_like(x)*(1.0 - ratio)
    #falseCase = tf.ones_like(x)*ratio

    #weights = tf.where(weights, trueCase, falseCase)
    #weights = tf.where(weights, trueCase, falseCase)
    #weighted_logits = weights*x

    
    ce = tf.nn.weighted_cross_entropy_with_logits(
        targets = targets,
        logits = logits,
        pos_weight = weight
    )
    '''
    ce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = targets,
        logits = logits
    )
    '''
    return tf.reduce_mean(ce)

class UNetModel(object):
    def __init__(self, input, output, is_training, weight):
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
            'transcription-unet',
            reuse = False
        )

        self.prediction = self.unet.output
        self.cost = sigmoid_xentropy(self.prediction, self.output, self._weight)
      
        self.train_op = self._optimizer.minimize(self.cost)

class UNet(object):
    def __init__(self, input, is_training, name, reuse = False):
        with tf.variable_scope(name, reuse = reuse):
            kernel_size = (5, 5)
            self.encoder = Encoder(input, kernel_size, is_training, reuse)
            self.decoder = Decoder(self.encoder, kernel_size, is_training, reuse)
            self.output = sigmoid(self.decoder.output)
