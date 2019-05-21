import os
import random
import sys
import time
import heapq
import tensorflow.python.platform
import numpy as np
from six.moves import xrange
import tensorflow as tf
import util
import model.gen_model as seq2seq_model
from tensorflow.python.platform import gfile

def read_data(config, source_path, target_path, max_size=None):
    data_set = [[] for _ in config.buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading disc_data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(util.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(config.buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, gen_config, forward_only, name_scope, initializer=None):
    """Create translation model and initialize or load parameters in session."""
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = seq2seq_model.Seq2SeqModel(gen_config,  name_scope=name_scope, forward_only=forward_only)
        gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
        ckpt = tf.train.get_checkpoint_state(gen_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Gen model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created Gen model with fresh parameters.")
            gen_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            session.run(tf.variables_initializer(gen_global_variables))
        return model

def prepare_data(gen_config):
    train_path = os.path.join(gen_config.train_dir, "chitchat.train")
    voc_file_path = [train_path+".answer", train_path+".query"]
    vocab_path = os.path.join(gen_config.train_dir, "vocab%d.all" % gen_config.vocab_size)
    util.create_vocabulary(vocab_path, voc_file_path, gen_config.vocab_size)
    vocab, rev_vocab = util.initialize_vocabulary(vocab_path)

    print("Preparing Chitchat gen_data in %s" % gen_config.train_dir)
    train_query, train_answer, dev_query, dev_answer = util.prepare_chitchat_data(
        gen_config.train_dir, vocab, gen_config.vocab_size)

    # Read disc_data into buckets and compute their sizes.
    print ("Reading development and training gen_data (limit: %d)."
               % gen_config.max_train_data_size)
    dev_set = read_data(gen_config, dev_query, dev_answer)
    train_set = read_data(gen_config, train_query, train_answer, gen_config.max_train_data_size)

    return vocab, rev_vocab, dev_set, train_set
