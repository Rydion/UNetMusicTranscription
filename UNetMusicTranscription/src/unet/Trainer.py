'''
Created on 2018-06-06
author: Adrian Hintze @Rydion
'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import logging
import shutil
import numpy as np
import tensorflow as tf

from src.unet.util import crop_to_shape, combine_img_prediction, save_image


class Trainer(object):
    verification_batch_size = 4

    @staticmethod
    def errorRate(predictions, labels):
        return 100.0 - (100.0*np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3))/(predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))

    def __init__(self, net, batch_size = 1, norm_grads = False, optimizer = 'momentum', opt_kwargs = {}):
        self.net = net
        self.batch_size = batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

    def train(self, dataProvider, outputPath, trainingIters = 10, epochs = 100, dropout = 0.75, displayStep = 1, restore = False, writeGraph = False, predictionPath = 'prediction'):
        savePath = os.path.join(outputPath, 'model.ckpt')
        if epochs == 0:
            return savePath

        init = self._initialize(trainingIters, outputPath, restore, predictionPath)
        with tf.Session() as sess:
            if writeGraph:
                tf.train.write_graph(sess.graph_def, outputPath, 'graph.pb', False)
            
            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(outputPath)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            x_test, y_test = dataProvider(self.verification_batch_size)
            pred_shape = self.store_prediction(sess, x_test, y_test, '_init')

            summaryWriter = tf.summary.FileWriter(outputPath, graph = sess.graph)
            logging.info('Start optimization')

            avgGradients = None
            for epoch in range(epochs):
                totalLoss = 0
                for step in range((epoch*trainingIters), ((epoch + 1)*trainingIters)):
                    batchX, batchY = dataProvider(self.batch_size)

                    # Run optimization op (backprop)
                    _, loss, lr, gradients = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                        feed_dict = {
                            self.net.x: batchX,
                            self.net.y: crop_to_shape(batchY, pred_shape),
                            self.net.keep_prob: dropout
                        }
                    )

                    if step % displayStep == 0:
                        self.outputMinibatchStats(
                            sess,
                            summaryWriter,
                            step,
                            batchX,
                            crop_to_shape(batchY, pred_shape)
                        )

                    totalLoss += loss

                self.outputEpochStats(epoch, totalLoss, trainingIters, lr)
                self.store_prediction(sess, x_test, y_test, 'epoch_%s' % epoch)

                savePath = self.net.save(sess, savePath)
            logging.info('Optimization Finished!')

            return savePath

    def store_prediction(self, sess, x_batch, y_batch, name):
        prediction = sess.run(
            self.net.predicter,
            feed_dict = {
                self.net.x: x_batch,
                self.net.y: y_batch,
                self.net.keep_prob: 1.0
            }
        )

        pred_shape = prediction.shape
        loss = sess.run(
            self.net.cost,
            feed_dict = {
                self.net.x: x_batch,
                self.net.y: crop_to_shape(y_batch, pred_shape),
                self.net.keep_prob: 1.0
            }
        )

        logging.info('Verification error= {:.1f}%, loss= {:.4f}'.format(Trainer.errorRate(prediction, crop_to_shape(y_batch, prediction.shape)), loss))
        img = combine_img_prediction(x_batch, y_batch, prediction)
        save_image(img, '%s/%s.jpg' % (self.prediction_path, name))
        return pred_shape

    def outputEpochStats(self, epoch, totalLoss, trainingIters, lr):
        logging.info('Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}'.format(epoch, (totalLoss/trainingIters), lr))

    def outputMinibatchStats(self, sess, summary_writer, step, x_batch, y_batch):
        summaryStr, loss, acc, predictions = sess.run([
                self.summary_op,
                self.net.cost,
                self.net.accuracy,
                self.net.predicter
            ],
            feed_dict = {
                self.net.x: x_batch,
                self.net.y: y_batch,
                self.net.keep_prob: 1.0
            }
        )
        summary_writer.add_summary(summaryStr, step)
        summary_writer.flush()
        logging.info('Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%'.format(step, loss, acc, Trainer.errorRate(predictions, y_batch)))

    def _getOptimizer(self, trainingIters, globalStep):
        if self.optimizer == 'momentum':
            learningRate = self.opt_kwargs.pop('learning_rate', 0.2)
            decayRate = self.opt_kwargs.pop('decay_rate', 0.95)
            momentum = self.opt_kwargs.pop('momentum', 0.2)

            self.learning_rate_node = tf.train.exponential_decay(
                learning_rate = learningRate,
                global_step = globalStep,
                decay_steps = trainingIters,
                decay_rate = decayRate,
                staircase = True
            )

            return tf.train.MomentumOptimizer(
                learning_rate = self.learning_rate_node,
                momentum = momentum,
                **self.opt_kwargs
            ) \
            .minimize(
                self.net.cost,
                global_step = globalStep
            )
        elif self.optimizer == 'adam':
            learningRate = self.opt_kwargs.pop('learning_rate', 0.001)
            self.learning_rate_node = tf.Variable(learningRate)

            return tf.train.AdamOptimizer(
                learning_rate = self.learning_rate_node,
                **self.opt_kwargs
            ) \
            .minimize(
                self.net.cost,
                global_step = globalStep
            )

        # TODO raise exception

    def _initialize(self, trainingIters, outputPath, restore, predictionPath):
        globalStep = tf.Variable(0)

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape = [len(self.net.gradients_node)]))

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._getOptimizer(trainingIters, globalStep)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        self.prediction_path = predictionPath
        absPredictionPath = os.path.abspath(self.prediction_path)
        outputPath = os.path.abspath(outputPath)

        if not restore:
            logging.info("Removing '{:}'".format(absPredictionPath))
            shutil.rmtree(absPredictionPath, ignore_errors = True)
            logging.info("Removing '{:}'".format(outputPath))
            shutil.rmtree(outputPath, ignore_errors = True)

        if not os.path.exists(absPredictionPath):
            logging.info("Allocating '{:}'".format(absPredictionPath))
            os.makedirs(absPredictionPath)

        if not os.path.exists(outputPath):
            logging.info("Allocating '{:}'".format(outputPath))
            os.makedirs(outputPath)

        return init


def get_image_summary(img, idx=0):
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V

def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

    return avg_gradients
