'''
created: 2018-06-06
edited: 2018-07-04
author: Adrian Hintze @Rydion
'''

import os
import logging
import shutil
import numpy as np
import tensorflow as tf

from unet.util import crop_to_shape, combine_img_prediction, save_image

class Trainer(object):
    verification_batch_size = 4

    @staticmethod
    def error_rate(predictions, labels):
        return 100.0 - (100.0*np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3))/(predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))

    def __init__(self, net, batch_size = 1, norm_grads = False, optimizer = 'momentum', opt_kwargs = {}):
        self.net = net
        self.batch_size = batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

    def train(self, data_provider, output_path, training_iters = 10, epochs = 100, dropout = 0.75, restore = False, write_graph = False, prediction_path = 'prediction'):
        save_path = os.path.join(output_path, 'model.ckpt')
        if epochs == 0:
            return save_path

        init = self._initialize(training_iters, output_path, restore, prediction_path)
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, 'graph.pb', False)
            
            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            x_test, y_test = data_provider(self.verification_batch_size)
            pred_shape = self.store_prediction(sess, x_test, y_test, '_init')

            summary_writer = tf.summary.FileWriter(output_path, graph = sess.graph)
            logging.info('Start optimization')

            avg_gradients = None
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch*training_iters), ((epoch + 1)*training_iters)):
                    x_batch, y_batch = data_provider(self.batch_size)

                    # Run optimization op (backprop)
                    _, loss, lr, gradients = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                        feed_dict = {
                            self.net.x: x_batch,
                            self.net.y: crop_to_shape(y_batch, pred_shape),
                            self.net.keep_prob: dropout
                        }
                    )

                    total_loss += loss

                self.store_prediction(sess, x_test, y_test, 'epoch_%s' % epoch)

                save_path = self.net.save(sess, save_path)

            logging.info('Optimization Finished!')
            return save_path

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

        logging.info('Verification error= {:.1f}%, loss= {:.4f}'.format(Trainer.error_rate(prediction, crop_to_shape(y_batch, prediction.shape)), loss))
        img = combine_img_prediction(x_batch, y_batch, prediction)
        save_image(img, '%s/%s.jpg' % (self.prediction_path, name))
        return pred_shape

    def _getOptimizer(self, training_iters, global_step):
        if self.optimizer == 'momentum':
            learning_rate = self.opt_kwargs.pop('learning_rate', 0.2)
            decay_rate = self.opt_kwargs.pop('decay_rate', 0.95)
            momentum = self.opt_kwargs.pop('momentum', 0.2)

            self.learning_rate_node = tf.train.exponential_decay(
                learning_rate = learning_rate,
                global_step = global_step,
                decay_steps = training_iters,
                decay_rate = decay_rate,
                staircase = True
            )

            return tf.train.MomentumOptimizer(
                learning_rate = self.learning_rate_node,
                momentum = momentum,
                **self.opt_kwargs
            )\
            .minimize(
                self.net.cost,
                global_step = global_step
            )
        elif self.optimizer == 'adam':
            learning_rate = self.opt_kwargs.pop('learning_rate', 0.001)
            self.learning_rate_node = tf.Variable(learning_rate)

            return tf.train.AdamOptimizer(
                learning_rate = self.learning_rate_node,
                **self.opt_kwargs
            )\
            .minimize(
                self.net.cost,
                global_step = global_step
            )

        # TODO raise exception

    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0)

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape = [len(self.net.gradients_node)]))

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._getOptimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors = True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors = True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init
