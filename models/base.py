import os
import tensorflow as tf


class Model(object):
    """Abstract object representing BiRNN model"""

    def __init__(self, c, embeds):
        self.c = c
        self.embeds = embeds
        self.init_global_step()
        self.init_cur_epoch()

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, os.path.join(self.c.ckp_dir, self.c.model_name),
                        self.global_step_tensor)
        print("Model saved")

    def load(self, sess):
        latest_ckp = tf.train.latest_checkpoint(os.path.join(
            self.c.ckp_dir, self.c.model_name))
        if latest_ckp:
            print("Loading model checkpoint {}...\n".format(latest_ckp))
            self.saver.restore(sess, latest_ckp)
            print("Model loaded")

    def init_cur_epoch(self):
        with tf.variable_scope("cur_epoch"):
            self.cur_epoch_tensor = tf.Variable(0, name="cur_epoch",
                                                trainable=False,
                                                dtype=tf.int32)
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor,
                                                        self.cur_epoch_tensor + 1)

    def init_global_step(self):
        """ DON'T forget to add the global step tensor to the trainer """
        with tf.variable_scope("global_step"):
            self.global_step_tensor = tf.Variable(0, name="global_step",
                                                  trainable=False,
                                                  dtype=tf.int32)

    def init_saver(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
