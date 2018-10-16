'''
author: Adrian Hintze @Rydion
'''

import tensorflow as tf

from unet.Encoder import Encoder
from unet.Decoder import Decoder

def tanh(inputs):
    return tf.nn.tanh(inputs)

def sigmoid(inputs):
    return tf.nn.sigmoid(inputs)

def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))

def l2_loss(x, y):
    return tf.nn.l2_loss(tf.abs(x - y))

def sigmoid_xentropy(x, y, weight):
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
    '''
    ce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = y,
        logits = x
    )
    return tf.reduce_mean(ce)
    '''
    ce = tf.nn.weighted_cross_entropy_with_logits(
        targets = y,
        logits = x,
        pos_weight = weight
    )
    
    return tf.reduce_mean(ce)

def normalize(x):
    min = tf.reduce_min(x)
    max = tf.reduce_max(x)
    return tf.div(
        tf.subtract(x, min),
        tf.subtract(max, min)
    )

class UNetModel(object):
    def __init__(self, input, output, is_training, weight):
        self.input = input
        self.output = output
        self.is_training = is_training
        self._weight = weight

        self.unet = UNet(
            self.input,
            self.is_training,
            'transcription-unet',
            reuse = False
        )

        self._prediction = self.unet.output
        self.prediction = sigmoid(self._prediction)

        #self.cost = l1_loss(self.prediction, self.output)
        #self.cost = l2_loss(self.prediction, self.output)
        self.cost = sigmoid_xentropy(self._prediction, self.output, self._weight)
        #self.cost = softmax_xentropy(self.prediction, self.output)

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
            self.output = self.decoder.output
