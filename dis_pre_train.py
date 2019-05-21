import config
from discriminator import *
import tensorflow as tf
import numpy as np
import os
import time
import random
from six.moves import xrange
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def dis_pre_train(config_disc, config_evl):
    config_evl.keep_prob = 1.0

    print("begin training")

    with tf.Session() as session:

        print("prepare_data")
        query_set, answer_set, gen_set = prepare_data(config_disc)

        train_bucket_sizes = [len(query_set[b]) for b in xrange(len(config_disc.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        for set in query_set:
            print("set length: ", len(set))

        model = create_model(session, config_disc, name_scope=config_disc.name_model)

        step_time, loss = 0.0, 0.0
        current_step = 0
        step_loss_summary = tf.Summary()

        train_step = config_disc.dis_pre_train_step
        while train_step>0:
            train_step -= 1
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()

            b_query, b_answer, b_gen = query_set[bucket_id], answer_set[bucket_id], gen_set[bucket_id]

            train_query, train_answer, train_labels = hier_get_batch(config_disc, len(b_query)-1, b_query, b_answer, b_gen)

            train_query = np.transpose(train_query)
            train_answer = np.transpose(train_answer)

            feed_dict = {}
            for i in xrange(config_disc.buckets[bucket_id][0]):
                feed_dict[model.query[i].name] = train_query[i]
            for i in xrange(config_disc.buckets[bucket_id][1]):
                feed_dict[model.answer[i].name] = train_answer[i]
            feed_dict[model.target.name] = train_labels

            fetches = [model.b_train_op[bucket_id], model.b_logits[bucket_id], model.b_loss[bucket_id], model.target]
            train_op, logits, step_loss, target = session.run(fetches, feed_dict)

            step_time += (time.time() - start_time) / config_disc.steps_per_checkpoint
            loss += step_loss /config_disc.steps_per_checkpoint
            current_step += 1

            if current_step % config_disc.steps_per_checkpoint == 0:

                disc_loss_value = step_loss_summary.value.add()
                disc_loss_value.tag = config_disc.name_loss
                disc_loss_value.simple_value = float(loss)

                print("logits shape: ", np.shape(logits))

                # softmax operation
                logits = np.transpose(softmax(np.transpose(logits)))

                reward = 0.0
                for logit, label in zip(logits, train_labels):
                    reward += logit[1]  # only for true probility
                reward = reward / len(train_labels)
                print("reward: ", reward)


                print("current_step: %d, step_loss: %.4f" %(current_step, step_loss))


                if current_step % (config_disc.steps_per_checkpoint * 3) == 0:
                    print("current_step: %d, save_model" % (current_step))
                    disc_ckpt_dir = os.path.abspath(os.path.join(config_disc.train_dir, "checkpoints"))
                    if not os.path.exists(disc_ckpt_dir):
                        os.makedirs(disc_ckpt_dir)
                    disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                    model.saver.save(session, disc_model_path, global_step=model.global_step)


                step_time, loss = 0.0, 0.0
                sys.stdout.flush()
                
if __name__ == "__main__":
    dis_pre_train(config.disc_config, config.disc_config)