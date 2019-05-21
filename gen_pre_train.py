import config
import os
import tensorflow as tf
from generator import *
import numpy as np
import time
import sys
import math
import random
from six.moves import xrange

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def gen_pre_train(gen_config):
    vocab, rev_vocab, dev_set, train_set = prepare_data(gen_config)
    for b_set in train_set:
        print("b_set: ", len(b_set))

    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.emb_dim))
        model = create_model(sess, gen_config, forward_only=False, name_scope=gen_config.name_model)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0

        gen_loss_summary = tf.Summary()

        train_step = gen_config.gen_pre_train_step
        while train_step>0:
            train_step -= 1
            # Choose a bucket according to disc_data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch(
                train_set, bucket_id, gen_config.batch_size)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=False)

            step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint
            loss += step_loss / gen_config.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % gen_config.steps_per_checkpoint == 0:

                bucket_value = gen_loss_summary.value.add()
                bucket_value.tag = gen_config.name_loss
                bucket_value.simple_value = float(loss)

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))

                # Save checkpoint and zero timer and loss.
                if current_step % (gen_config.steps_per_checkpoint * 3) == 0:
                    print("current_step: %d, save model" %(current_step))
                    gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
                    if not os.path.exists(gen_ckpt_dir):
                        os.makedirs(gen_ckpt_dir)
                    checkpoint_path = os.path.join(gen_ckpt_dir, "chitchat.model")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)


                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

if __name__ == "__main__":
    gen_pre_train(config.gen_config)