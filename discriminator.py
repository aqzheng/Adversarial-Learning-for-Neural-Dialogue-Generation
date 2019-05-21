import tensorflow as tf
import numpy as np
import os
import time
import random
from six.moves import xrange
from model.dis_model import Hier_rnn_model
import util

from tensorflow.python.platform import gfile
import sys

def hier_read_data(config, query_path, answer_path, gen_path):
    query_set = [[] for _ in config.buckets]
    answer_set = [[] for _ in config.buckets]
    gen_set = [[] for _ in config.buckets]
    with gfile.GFile(query_path, mode="r") as query_file:
        with gfile.GFile(answer_path, mode="r") as answer_file:
            with gfile.GFile(gen_path, mode="r") as gen_file:
                query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()
                counter = 0
                while query and answer and gen:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading disc_data line %d" % counter)
                    query = [int(id) for id in query.strip().split()]
                    answer = [int(id) for id in answer.strip().split()]
                    gen = [int(id) for id in gen.strip().split()]
                    for i, (query_size, answer_size) in enumerate(config.buckets):
                        if len(query) <= query_size and len(answer) <= answer_size and len(gen) <= answer_size:
                            query = query[:query_size] + [util.PAD_ID] * (query_size - len(query) if query_size > len(query) else 0)
                            query_set[i].append(query)
                            answer = answer[:answer_size] + [util.PAD_ID] * (answer_size - len(answer) if answer_size > len(answer) else 0)
                            answer_set[i].append(answer)
                            gen = gen[:answer_size] + [util.PAD_ID] * (answer_size - len(gen) if answer_size > len(gen) else 0)
                            gen_set[i].append(gen)
                    query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()

    return query_set, answer_set, gen_set


def hier_get_batch(config, max_set, query_set, answer_set, gen_set):
    batch_size = config.batch_size
    if batch_size % 2 == 1:
        return IOError("Error")
    train_query = []
    train_answer = []
    train_labels = []
    half_size = int(batch_size / 2)
    for _ in range(half_size):
        index = random.randint(0, max_set)
        train_query.append(query_set[index])
        train_answer.append(answer_set[index])
        train_labels.append(1)
        train_query.append(query_set[index])
        train_answer.append(gen_set[index])
        train_labels.append(0)
    return train_query, train_answer, train_labels


def create_model(sess, config, name_scope, initializer=None):
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = Hier_rnn_model(config=config, name_scope=name_scope)
        disc_ckpt_dir = os.path.abspath(os.path.join(config.train_dir, "checkpoints"))
        ckpt = tf.train.get_checkpoint_state(disc_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Hier Disc model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created Hier Disc model with fresh parameters.")
            disc_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            sess.run(tf.variables_initializer(disc_global_variables))
        return model


def prepare_data(config):
    train_path = os.path.join(config.train_dir, "train")
    voc_file_path = [train_path + ".query", train_path + ".answer", train_path + ".gen"]
    vocab_path = os.path.join(config.train_dir, "vocab%d.all" % config.vocab_size)
    util.create_vocabulary(vocab_path, voc_file_path, config.vocab_size)
    vocab, rev_vocab = util.initialize_vocabulary(vocab_path)

    print("Preparing train disc_data in %s" % config.train_dir)
    train_query_path, train_answer_path, train_gen_path, dev_query_path, dev_answer_path, dev_gen_path = \
        util.hier_prepare_disc_data(config.train_dir, vocab, config.vocab_size)
    query_set, answer_set, gen_set = hier_read_data(config, train_query_path, train_answer_path, train_gen_path)
    return query_set, answer_set, gen_set


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob
