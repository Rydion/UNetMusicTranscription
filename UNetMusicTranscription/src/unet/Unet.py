'''
created on 2018-05-31
author: Adrian Hintze @Rydion
'''

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import tensorflow as tf

from collections import OrderedDict


class Unet:
    @staticmethod
    def bias(shape):
        return tf.Variable(tf.constant(0.1, shape = shape))

    @staticmethod
    def weight(shape, stddev = 0.1):
        return tf.Variable(tf.truncated_normal(shape, stddev = stddev))

    @staticmethod
    def weightDeconv(shape, stddev = 0.1):
        return tf.Variable(tf.truncated_normal(shape, stddev = stddev))

    @staticmethod
    def conv2d(x, W, keep_prob):
        layer = tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')
        return tf.nn.dropout(layer, keep_prob)
    
    @staticmethod
    def deconv2d(x, W, stride):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, stride, stride, 1], padding = 'VALID')

    @staticmethod
    def maxPool(x, n):
        return tf.nn.max_pool(x, ksize = [1, n, n, 1], strides = [1, n, n, 1], padding = 'VALID')

    @staticmethod
    def crop_and_concat(x1, x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        offsets = [0, (x1_shape[1] - x2_shape[1])//2, (x1_shape[2] - x2_shape[2])//2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

    @staticmethod
    def pixelwise_softmax(outputmap):
        exponentialmap = tf.exp(outputmap)
        exp_sum = tf.reduce_sum(exponentialmap, 3, keepdims = True)
        exp_sum_tensor = tf.tile(exp_sum, tf.stack([1, 1, 1, tf.shape(outputmap)[3]]))
        return tf.div(exponentialmap, exp_sum_tensor)

    @staticmethod
    def calc_cross_entropy(y, outputmap):
        return -tf.reduce_mean(y*tf.log(tf.clip_by_value(outputmap, 1e-10, 1.0)), name = 'cross_entropy')

    def __init__(self, num_channels = 3, num_classes = 2, cost = 'cross_entropy', cost_kwargs = {},  **kwargs):
        self._num_channels = num_channels
        self._num_classes = num_classes
        
        self.x = tf.placeholder('float', shape = [None, None, None, self._num_channels])
        self.y = tf.placeholder('float', shape = [None, None, None, self._num_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        
        logits, self._variables, self._offset = self._initUnet(**kwargs)

        self.cost = self._getCost(logits, cost, cost_kwargs)
        self.gradients_node = tf.gradients(self.cost, self._variables)
        self.cross_entropy = tf.reduce_mean(Unet.calc_cross_entropy(tf.reshape(self.y, [-1, self._num_classes]), tf.reshape(Unet.pixelwise_softmax(logits), [-1, self._num_classes])))

        self.predicter = Unet.pixelwise_softmax(logits)
        self._correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))

    def save(self, sess, modelPath):
        saver = tf.train.Saver()
        savePath = saver.save(sess, modelPath)
        return savePath

    def _initUnet(self, layers = 3, featuresRoot = 16, filterSize = 3, poolSize = 2):
        nx = tf.shape(self.x)[1]
        ny = tf.shape(self.x)[2]
        input = tf.reshape(self.x, tf.stack([-1, nx, ny, self._num_channels]))
        batchSize = tf.shape(input)[0]
 
        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        downHConvs = OrderedDict()
        upHConvs = OrderedDict()
    
        # down layers
        inSize = 1000
        size = inSize
        for layer in range(0, layers):
            features = 2**layer*featuresRoot
            stddev = np.sqrt(2/(filterSize**2*features))
            if layer == 0:
                w1 = Unet.weight([filterSize, filterSize, self._num_channels, features], stddev)
            else:
                w1 = Unet.weight([filterSize, filterSize, features//2, features], stddev)
            
            w2 = Unet.weight([filterSize, filterSize, features, features], stddev)
            b1 = Unet.bias([features])
            b2 = Unet.bias([features])
        
            conv1 = Unet.conv2d(input, w1, self.keep_prob)
            tmpHConv = tf.nn.relu(conv1 + b1)
            conv2 = Unet.conv2d(tmpHConv, w2, self.keep_prob)
            downHConvs[layer] = tf.nn.relu(conv2 + b2)
        
            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size -= 4
            if layer < layers - 1:
                pools[layer] = Unet.maxPool(downHConvs[layer], poolSize)
                input = pools[layer]
                size /= 2
        
        input = downHConvs[layers - 1]
        
        # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2**(layer + 1)*featuresRoot
            stddev = np.sqrt(2/(filterSize**2*features))
        
            wd = Unet.weightDeconv([poolSize, poolSize, features//2, features], stddev)
            bd = Unet.bias([features//2])
            hDeconv = tf.nn.relu(Unet.deconv2d(input, wd, poolSize) + bd)
            hDeconvConcat = Unet.crop_and_concat(downHConvs[layer], hDeconv)
            deconv[layer] = hDeconvConcat
        
            w1 = Unet.weight([filterSize, filterSize, features, features//2], stddev)
            w2 = Unet.weight([filterSize, filterSize, features//2, features//2], stddev)
            b1 = Unet.bias([features//2])
            b2 = Unet.bias([features//2])
        
            conv1 = Unet.conv2d(hDeconvConcat, w1, self.keep_prob)
            hConv = tf.nn.relu(conv1 + b1)
            conv2 = Unet.conv2d(hConv, w2, self.keep_prob)
            input = tf.nn.relu(conv2 + b2)
            upHConvs[layer] = input

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size *= 2
            size -= 4

        # Output Map
        weight = Unet.weight([1, 1, featuresRoot, self._num_classes], stddev)
        bias = Unet.bias([self._num_classes])
        conv = Unet.conv2d(input, weight, tf.constant(1.0))
        outputMap = tf.nn.relu(conv + bias)
        upHConvs['out'] = outputMap
            
        variables = []
        for w1, w2 in weights:
            variables.append(w1)
            variables.append(w2)
        
        for b1, b2 in biases:
            variables.append(b1)
            variables.append(b2)

        return outputMap, variables, int(inSize - size)

    def _getCost(self, logits, costName, costKwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        flatLogits = tf.reshape(logits, [-1, self._num_classes])
        flatLabels = tf.reshape(self.y, [-1, self._num_classes])
        if costName == 'cross_entropy':
            classWeights = costKwargs.pop('class_weights', None)

            if classWeights is not None:
                classWeights = tf.constant(np.array(classWeights, dtype = np.float32))

                weightMap = tf.multiply(flatLabels, classWeights)
                weightMap = tf.reduce_sum(weightMap, axis = 1)

                lossMap = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits = flatLogits,
                    labels = flatLabels
                )
                weightedLoss = tf.multiply(lossMap, weightMap)

                loss = tf.reduce_mean(weightedLoss)

            else:
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits = flatLogits,
                        labels = flatLabels
                    )
                )
        elif costName == 'dice_coefficient':
            eps = 1e-5
            prediction = Unet.pixelwise_softmax(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2*intersection/(union))

        else:
            raise ValueError('Unknown cost function: ' % costName)

        regularizer = costKwargs.pop('regularizer', None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += regularizer*regularizers

        return loss
