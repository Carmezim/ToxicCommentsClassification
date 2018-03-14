import tensorflow as tf
from tensorflow.contrib import rnn
from models.base import Model


class BiLSTM(Model):
    def __init__(self, c, embeds):
        super(BiLSTM, self).__init__(c, embeds)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.int32, [self.c.b, self.c.max_len], "x")
        self.y = tf.placeholder(tf.float32, [self.c.b, self.c.n_classes], "y")

        # weights initializer
        self.initializer = tf.random_normal_initializer(stddev=0.1)

        # input length for static LSTM
        # self.input_len = tf.placeholder(tf.int32, [self.c.b])

        # forward and backward cells
        self.f_c = rnn.LSTMCell(self.c.n_hidden, forget_bias=1.0)
        self.f_c = rnn.DropoutWrapper(cell=self.f_c,
                                      output_keep_prob=self.c.o_p)
        self.b_c = rnn.LSTMCell(self.c.n_hidden, forget_bias=1.0)
        self.b_c = rnn.DropoutWrapper(cell=self.b_c,
                                      output_keep_prob=self.c.o_p)
        # weights and biases
        self.w = tf.get_variable("w", [2 * self.c.n_hidden, self.c.n_classes],
                                 initializer=self.initializer)
        self.b = tf.get_variable("b", [self.c.n_classes])

        # pre-trained embeddings
        self.word_vectors = tf.nn.embedding_lookup(self.embeds,
                                                   tf.transpose(self.x))
        self.word_vectors = tf.cast(self.word_vectors, tf.float32)
        self.word_vectors = tf.unstack(self.word_vectors)

        # RNN
        self.output, _, _ = rnn.static_bidirectional_rnn(cell_fw=self.f_c,
                                                 cell_bw=self.b_c,
                                                 inputs=self.word_vectors,
                                                 # sequence_length=[self.c.b] *
                                                 # 				self.c.max_len,
                                                 dtype=tf.float32)

        # RNN prediction
        self.scores = tf.matmul(self.output[-1], self.w) + self.b

        # Loss
        with tf.variable_scope("loss"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y, logits=self.scores))

            # learning rate decay
            self.lr = tf.train.exponential_decay(learning_rate=self.c.lr,
                                         global_step=self.global_step_tensor,
                                         decay_steps=self.c.lr_decay_steps,
                                         decay_rate=self.c.lr_decay,
                                         staircase=False)

            # gradient clipping
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

            self.correct_scores = tf.equal(tf.argmax(self.scores, 1),
                                           tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_scores,
                                                   tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.c.max_to_keep)
