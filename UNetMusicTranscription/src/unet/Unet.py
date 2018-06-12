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
    def weight_deconv(shape, stddev = 0.1):
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
    def max_pool(x, n):
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
        
        logits, self._variables, self._offset = self._init_unet(**kwargs)

        self.cost = self._getCost(logits, cost, cost_kwargs)
        self.gradients_node = tf.gradients(self.cost, self._variables)
        self.cross_entropy = tf.reduce_mean(Unet.calc_cross_entropy(tf.reshape(self.y, [-1, self._num_classes]), tf.reshape(Unet.pixelwise_softmax(logits), [-1, self._num_classes])))

        self.predicter = Unet.pixelwise_softmax(logits)
        self._correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))

    def save(self, sess, model_path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def _init_unet(self, layers = 3, features_root = 16, filter_size = 3, pool_size = 2):
        nx = tf.shape(self.x)[1]
        ny = tf.shape(self.x)[2]
        input = tf.reshape(self.x, tf.stack([-1, nx, ny, self._num_channels]))
        batch_size = tf.shape(input)[0]
 
        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        down_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
    
        # down layers
        in_size = 1000
        size = in_size
        for layer in range(0, layers):
            features = 2**layer*features_root
            stddev = np.sqrt(2/(filter_size**2*features))
            if layer == 0:
                w1 = Unet.weight([filter_size, filter_size, self._num_channels, features], stddev)
            else:
                w1 = Unet.weight([filter_size, filter_size, features//2, features], stddev)
            
            w2 = Unet.weight([filter_size, filter_size, features, features], stddev)
            b1 = Unet.bias([features])
            b2 = Unet.bias([features])
        
            conv1 = Unet.conv2d(input, w1, self.keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = Unet.conv2d(tmp_h_conv, w2, self.keep_prob)
            down_h_convs[layer] = tf.nn.relu(conv2 + b2)
        
            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size -= 4
            if layer < layers - 1:
                pools[layer] = Unet.max_pool(down_h_convs[layer], pool_size)
                input = pools[layer]
                size /= 2
        
        input = down_h_convs[layers - 1]
        
        # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2**(layer + 1)*features_root
            stddev = np.sqrt(2/(filter_size**2*features))
        
            wd = Unet.weight_deconv([pool_size, pool_size, features//2, features], stddev)
            bd = Unet.bias([features//2])
            h_deconv = tf.nn.relu(Unet.deconv2d(input, wd, pool_size) + bd)
            h_deconv_concat = Unet.crop_and_concat(down_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat
        
            w1 = Unet.weight([filter_size, filter_size, features, features//2], stddev)
            w2 = Unet.weight([filter_size, filter_size, features//2, features//2], stddev)
            b1 = Unet.bias([features//2])
            b2 = Unet.bias([features//2])
        
            conv1 = Unet.conv2d(h_deconv_concat, w1, self.keep_prob)
            hConv = tf.nn.relu(conv1 + b1)
            conv2 = Unet.conv2d(hConv, w2, self.keep_prob)
            input = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = input

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size *= 2
            size -= 4

        # Output Map
        weight = Unet.weight([1, 1, features_root, self._num_classes], stddev)
        bias = Unet.bias([self._num_classes])
        conv = Unet.conv2d(input, weight, tf.constant(1.0))
        outputmap = tf.nn.relu(conv + bias)
        up_h_convs['out'] = outputmap
            
        variables = []
        for w1, w2 in weights:
            variables.append(w1)
            variables.append(w2)
        
        for b1, b2 in biases:
            variables.append(b1)
            variables.append(b2)

        return outputmap, variables, int(in_size - size)

    def _getCost(self, logits, cost_name, cost_kwargs):
        flat_logits = tf.reshape(logits, [-1, self._num_classes])
        flat_labels = tf.reshape(self.y, [-1, self._num_classes])
        if cost_name == 'cross_entropy':
            class_weights = cost_kwargs.pop('class_weights', None)

            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype = np.float32))

                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis = 1)

                loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits = flat_logits,
                    labels = flat_labels
                )
                weighted_loss = tf.multiply(loss_map, weight_map)
                loss = tf.reduce_mean(weighted_loss)
            else:
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits = flat_logits,
                        labels = flat_labels
                    )
                )
        elif cost_name == 'dice_coefficient':
            eps = 1e-5
            prediction = Unet.pixelwise_softmax(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2*intersection/(union))

        else:
            raise ValueError('Unknown cost function: ' % cost_name)

        regularizer = cost_kwargs.pop('regularizer', None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += regularizer*regularizers

        return loss
