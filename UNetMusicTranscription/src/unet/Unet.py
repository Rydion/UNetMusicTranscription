'''
created on 2018-05-31
author: Adrian Hintze @Rydion
'''

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
    def weightDevonc(shape, stddev = 0.1):
        return tf.Variable(tf.truncated_normal(shape, stddev = stddev))

    @staticmethod
    def conv2d(x, W, keepProb):
        layer = tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')
        return tf.nn.dropout(layer, keepProb)
    
    @staticmethod
    def deconv2d(x, W, stride):
        xShape = tf.shape(x)
        output_shape = tf.stack([xShape[0], xShape[1]*2, xShape[2]*2, xShape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, stride, stride, 1], padding = 'VALID')

    @staticmethod
    def maxPool(x, n):
        return tf.nn.max_pool(x, ksize = [1, n, n, 1], strides = [1, n, n, 1], padding = 'VALID')

    @staticmethod
    def cropAndConcat(x1, x2):
        x1Shape = tf.shape(x1)
        x2Shape = tf.shape(x2)
        offsets = [0, (x1Shape[1] - x2Shape[1])//2, (x1Shape[2] - x2Shape[2])//2, 0]
        size = [-1, x2Shape[1], x2Shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

    @staticmethod
    def _initUnet(x, numChannels, numClasses, keepProb, layers = 3, features_root = 16, filter_size = 3, pool_size = 2):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, numChannels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]
 
        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
    
        in_size = 1000
        size = in_size
        for layer in range(0, layers):
            features = 2**layer*features_root
            stddev = np.sqrt(2/(filter_size**2*features))
            if layer == 0:
                w1 = Unet.weight([filter_size, filter_size, numChannels, features], stddev)
            else:
                w1 = Unet.weight([filter_size, filter_size, features//2, features], stddev)
            
            w2 = Unet.weight([filter_size, filter_size, features, features], stddev)
            b1 = Unet.bias([features])
            b2 = Unet.bias([features])
        
            conv1 = Unet.conv2d(in_node, w1, keepProb)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = Unet.conv2d(tmp_h_conv, w2, keepProb)
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
        
            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size -= 4
            if layer < layers-1:
                pools[layer] = Unet.maxPool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2
        
        in_node = dw_h_convs[layers-1]
        
        # up layers
        for layer in range(layers-2, -1, -1):
            features = 2**(layer+1)*features_root
            stddev = np.sqrt(2/(filter_size**2*features))
        
            wd = Unet.weightDevonc([pool_size, pool_size, features//2, features], stddev)
            bd = Unet.bias([features//2])
            h_deconv = tf.nn.relu(Unet.deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = Unet.cropAndConcat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat
        
            w1 = Unet.weight([filter_size, filter_size, features, features//2], stddev)
            w2 = Unet.weight([filter_size, filter_size, features//2, features//2], stddev)
            b1 = Unet.bias([features//2])
            b2 = Unet.bias([features//2])
        
            conv1 = Unet.conv2d(h_deconv_concat, w1, keepProb)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = Unet.conv2d(h_conv, w2, keepProb)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size *= 2
            size -= 4

        # Output Map
        weight = Unet.weight([1, 1, features_root, numClasses], stddev)
        bias = Unet.bias([numClasses])
        conv = Unet.conv2d(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        up_h_convs['out'] = output_map
            
        variables = []
        for w1, w2 in weights:
            variables.append(w1)
            variables.append(w2)
        
        for b1, b2 in biases:
            variables.append(b1)
            variables.append(b2)

        return output_map, variables, int(in_size - size)


    def __init__(self, numChannels = 3, numClasses = 2, **kwargs):
        self._numChannels = numChannels
        self._numClasses = numClasses
        
        self._x = tf.placeholder('float', shape = [None, None, None, self._numChannels])
        self._y = tf.placeholder('float', shape = [None, None, None, self._numClasses])
        self._keepProb = tf.placeholder(tf.float32)
        
        logits, self._variables, self._offset = Unet._initUnet(self._x, self._numChannels, self._numClasses, self._keepProb, **kwargs)


    _numChannels = 0
    _numClasses = 0
    _x = None
    _y = None
    _keepProb = None
    _variables = None
    _offset = 0
