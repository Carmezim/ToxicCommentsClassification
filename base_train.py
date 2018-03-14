import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm


class BaseTrain:
    def __init__(self, sess, model, data, c, logger):
        self.sess = sess
        self.data = data
        self.c = c
        self.model = model
        self.logger = logger
        self.init = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(
                self.sess).item(), self.c.n_epochs):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """Loop over number of iterations, add train steps and summaries"""
        raise NotImplementedError

    def train_step(self):
        """Run session and return stuff to summary"""
        raise NotImplementedError
