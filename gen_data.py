import config
import tensorflow as tf
from generator import *
import random
import numpy as np
from six.moves import xrange
import util

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def gen_data(gen_config):
    vocab, rev_vocab, dev_set, train_set = prepare_data(gen_config)

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    with tf.Session() as sess:
        model = create_model(sess, gen_config, forward_only=True, name_scope=gen_config.name_model)

        disc_train_query = open("dis_data/train.query", "w", encoding='utf-8')
        disc_train_answer = open("dis_data/train.answer", "w", encoding='utf-8')
        disc_train_gen = open("dis_data/train.gen", "w", encoding='utf-8')

        disc_dev_query = open("dis_data/dev.query", "w", encoding='utf-8')
        disc_dev_answer = open("dis_data/dev.answer", "w", encoding='utf-8')
        disc_dev_gen = open("dis_data/dev.gen", "w", encoding='utf-8')

        num_step = 0
        while num_step < 10000:
            print("generating num_step: ", num_step)
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = \
                model.get_batch(train_set, bucket_id, gen_config.batch_size)

            _, _, out_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                          forward_only=True)

            tokens = []
            resps = []
            for seq in out_logits:
                token = []
                for t in seq:
                    token.append(int(np.argmax(t, axis=0)))
                tokens.append(token)
            tokens_t = []
            for col in range(len(tokens[0])):
                tokens_t.append([tokens[row][col] for row in range(len(tokens))])

            for seq in tokens_t:
                if util.EOS_ID in seq:
                    resps.append(seq[:seq.index(util.EOS_ID)][:gen_config.buckets[bucket_id][1]])
                else:
                    resps.append(seq[:gen_config.buckets[bucket_id][1]])

            if num_step % 100 == 0:
                for query, answer, resp in zip(batch_source_encoder, batch_source_decoder, resps):

                    answer_str = " ".join([str(rev_vocab[an]) for an in answer][:-1])
                    disc_dev_answer.write(answer_str)
                    disc_dev_answer.write("\n")

                    query_str = " ".join([str(rev_vocab[qu]) for qu in query])
                    disc_dev_query.write(query_str)
                    disc_dev_query.write("\n")

                    resp_str = " ".join([tf.compat.as_str(rev_vocab[output]) for output in resp])

                    disc_dev_gen.write(resp_str)
                    disc_dev_gen.write("\n")
            else:
                for query, answer, resp in zip(batch_source_encoder, batch_source_decoder, resps):

                    answer_str = " ".join([str(rev_vocab[an]) for an in answer][:-1])
                    disc_train_answer.write(answer_str)
                    disc_train_answer.write("\n")

                    query_str = " ".join([str(rev_vocab[qu]) for qu in query])
                    disc_train_query.write(query_str)
                    disc_train_query.write("\n")

                    resp_str = " ".join([tf.compat.as_str(rev_vocab[output]) for output in resp])

                    disc_train_gen.write(resp_str)
                    disc_train_gen.write("\n")

            num_step += 1

        disc_train_gen.close()
        disc_train_query.close()
        disc_train_answer.close()
        disc_dev_gen.close()
        disc_dev_query.close()
        disc_dev_answer.close()
    pass

if __name__ == "__main__":
    gen_data(config.gen_config)