import tensorflow as tf
import numpy as np
from six.moves import xrange


class Hier_rnn_model(object):
    def __init__(self, config, name_scope, dtype=tf.float32):
        emb_dim = config.embed_dim
        num_layers = config.num_layers
        vocab_size = config.vocab_size
        num_class = config.num_class
        buckets = config.buckets
        self.lr = config.lr
        self.global_step = tf.Variable(initial_value=0, trainable=False)

        self.query = []
        self.answer = []
        for i in range(buckets[-1][0]):
            self.query.append(tf.placeholder(dtype=tf.int32, shape=[None], name="query{0}".format(i)))
        for i in xrange(buckets[-1][1]):
            self.answer.append(tf.placeholder(dtype=tf.int32, shape=[None], name="answer{0}".format(i)))

        self.target = tf.placeholder(dtype=tf.int64, shape=[None], name="target")

        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(emb_dim)
        encoder_mutil = tf.contrib.rnn.MultiRNNCell([encoder_cell] * num_layers)
        encoder_emb = tf.contrib.rnn.EmbeddingWrapper(encoder_mutil, embedding_classes=vocab_size, embedding_size=emb_dim)

        context_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=emb_dim)
        context_multi = tf.contrib.rnn.MultiRNNCell([context_cell] * num_layers)

        self.b_query_state = []
        self.b_answer_state = []
        self.b_state = []
        self.b_logits = []
        self.b_loss = []
        self.b_train_op = []
        for i, bucket in enumerate(buckets):
            with tf.variable_scope(name_or_scope="Hier_RNN_encoder", reuse=True if i > 0 else None) as var_scope:
                query_output, query_state = tf.contrib.rnn.static_rnn(encoder_emb, inputs=self.query[:bucket[0]], dtype=tf.float32)
                var_scope.reuse_variables()
                answer_output, answer_state = tf.contrib.rnn.static_rnn(encoder_emb, inputs=self.answer[:bucket[1]], dtype=tf.float32)
                self.b_query_state.append(query_state)
                self.b_answer_state.append(answer_state)
                context_input = [query_state[-1][1], answer_state[-1][1]]

            with tf.variable_scope(name_or_scope="Hier_RNN_context", reuse=True if i > 0 else None):
                output, state = tf.contrib.rnn.static_rnn(context_multi, context_input, dtype=tf.float32)
                self.b_state.append(state)
                top_state = state[-1][1]  # [batch_size, emb_dim]

            with tf.variable_scope("Softmax_layer_and_output", reuse=True if i > 0 else None):
                softmax_w = tf.get_variable("softmax_w", [emb_dim, num_class], dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b", [num_class], dtype=tf.float32)
                logits = tf.matmul(top_state, softmax_w) + softmax_b
                self.b_logits.append(logits)

            with tf.name_scope("loss"):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target)
                mean_loss = tf.reduce_mean(loss)
                self.b_loss.append(mean_loss)

            with tf.name_scope("gradient_descent"):
                disc_params = [var for var in tf.trainable_variables() if name_scope in var.name]
                grads, norm = tf.clip_by_global_norm(tf.gradients(mean_loss, disc_params), config.max_grad_norm)
                #optimizer = tf.train.GradientDescentOptimizer(self.lr)
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                train_op = optimizer.apply_gradients(zip(grads, disc_params), global_step=self.global_step)
                self.b_train_op.append(train_op)

        all_variables = [v for v in tf.global_variables() if name_scope in v.name]
        self.saver = tf.train.Saver(all_variables)
