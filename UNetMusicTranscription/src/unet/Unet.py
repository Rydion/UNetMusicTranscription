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
    def conv2d(x, W, keepProb):
        layer = tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')
        return tf.nn.dropout(layer, keepProb)
    
    @staticmethod
    def deconv2d(x, W, stride):
        xShape = tf.shape(x)
        outputShape = tf.stack([xShape[0], xShape[1]*2, xShape[2]*2, xShape[3]//2])
        return tf.nn.conv2d_transpose(x, W, outputShape, strides = [1, stride, stride, 1], padding = 'VALID')

    @staticmethod
    def maxPool(x, n):
        return tf.nn.max_pool(x, ksize = [1, n, n, 1], strides = [1, n, n, 1], padding = 'VALID')

    @staticmethod
    def cropAndConcat(x1, x2):
        x1Shape = tf.shape(x1)
        x2Shape = tf.shape(x2)
        offsets = [0, (x1Shape[1] - x2Shape[1])//2, (x1Shape[2] - x2Shape[2])//2, 0]
        size = [-1, x2Shape[1], x2Shape[2], -1]
        x1Crop = tf.slice(x1, offsets, size)
        return tf.concat([x1Crop, x2], 3)

    @staticmethod
    def pixelWiseSoftmax(outputMap):
        exponentialMap = tf.exp(outputMap)
        sumExp = tf.reduce_sum(exponentialMap, 3, keepdims = True)
        tensorSumExp = tf.tile(sumExp, tf.stack([1, 1, 1, tf.shape(outputMap)[3]]))
        return tf.div(exponentialMap, tensorSumExp)

    @staticmethod
    def calcCrossEntropy(y, outputMap):
        return -tf.reduce_mean(y*tf.log(tf.clip_by_value(outputMap, 1e-10, 1.0)), name = 'cross_entropy')

    def __init__(self, numChannels = 3, numClasses = 2, cost = 'cross_entropy', costKwargs = {},  **kwargs):
        self._numChannels = numChannels
        self._numClasses = numClasses
        
        self.x = tf.placeholder('float', shape = [None, None, None, self._numChannels])
        self.y = tf.placeholder('float', shape = [None, None, None, self._numClasses])
        self.keepProb = tf.placeholder(tf.float32)
        
        logits, self._variables, self._offset = self._initUnet(**kwargs)

        self.cost = self._getCost(logits, cost, costKwargs)
        self.gradientsNode = tf.gradients(self.cost, self._variables)
        self.crossEntropy = tf.reduce_mean(Unet.calcCrossEntropy(tf.reshape(self.y, [-1, self._numClasses]), tf.reshape(Unet.pixelWiseSoftmax(logits), [-1, self._numClasses])))

        self.predicter = Unet.pixelWiseSoftmax(logits)
        self._correctPred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self._correctPred, tf.float32))

    def save(self, sess, modelPath):
        saver = tf.train.Saver()
        savePath = saver.save(sess, modelPath)
        return savePath

    def _initUnet(self, layers = 3, featuresRoot = 16, filterSize = 3, poolSize = 2):
        nx = tf.shape(self.x)[1]
        ny = tf.shape(self.x)[2]
        input = tf.reshape(self.x, tf.stack([-1, nx, ny, self._numChannels]))
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
                w1 = Unet.weight([filterSize, filterSize, self._numChannels, features], stddev)
            else:
                w1 = Unet.weight([filterSize, filterSize, features//2, features], stddev)
            
            w2 = Unet.weight([filterSize, filterSize, features, features], stddev)
            b1 = Unet.bias([features])
            b2 = Unet.bias([features])
        
            conv1 = Unet.conv2d(input, w1, self.keepProb)
            tmpHConv = tf.nn.relu(conv1 + b1)
            conv2 = Unet.conv2d(tmpHConv, w2, self.keepProb)
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
            hDeconvConcat = Unet.cropAndConcat(downHConvs[layer], hDeconv)
            deconv[layer] = hDeconvConcat
        
            w1 = Unet.weight([filterSize, filterSize, features, features//2], stddev)
            w2 = Unet.weight([filterSize, filterSize, features//2, features//2], stddev)
            b1 = Unet.bias([features//2])
            b2 = Unet.bias([features//2])
        
            conv1 = Unet.conv2d(hDeconvConcat, w1, self.keepProb)
            hConv = tf.nn.relu(conv1 + b1)
            conv2 = Unet.conv2d(hConv, w2, self.keepProb)
            input = tf.nn.relu(conv2 + b2)
            upHConvs[layer] = input

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size *= 2
            size -= 4

        # Output Map
        weight = Unet.weight([1, 1, featuresRoot, self._numClasses], stddev)
        bias = Unet.bias([self._numClasses])
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

        flatLogits = tf.reshape(logits, [-1, self._numClasses])
        flatLabels = tf.reshape(self.y, [-1, self._numClasses])
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
            prediction = Unet.pixelWiseSoftmax(logits)
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


    x = None
    y = None
    keepProb = 0
    cost = 0
    crossEntropy = 0
    gradientsNode = None
    predicter = None

    _correctPred = 0
    _numChannels = 0
    _numClasses = 0
    _variables = None
    _offset = 0
