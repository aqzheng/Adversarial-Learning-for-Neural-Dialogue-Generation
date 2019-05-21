import config
import os
import sys
import numpy as np
import tensorflow as tf
import util
from generator import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def test(gen_config):
    with tf.Session() as sess:
        model = create_model(sess, gen_config, forward_only=True, name_scope=gen_config.name_model)
        model.batch_size = 1

        train_path = os.path.join(gen_config.train_dir, "chitchat.train")
        voc_file_path = [train_path + ".answer", train_path + ".query"]
        vocab_path = os.path.join(gen_config.train_dir, "vocab%d.all" % gen_config.vocab_size)
        util.create_vocabulary(vocab_path, voc_file_path, gen_config.vocab_size)
        vocab, rev_vocab = util.initialize_vocabulary(vocab_path)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            token_ids = util.sentence_to_token_ids(tf.compat.as_str_any(sentence), vocab)
            # print("token_id: ", token_ids)
            bucket_id = len(gen_config.buckets) - 1
            for i, bucket in enumerate(gen_config.buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            # else:
                # print("Sentence truncated: %s", sentence)

            encoder_inputs, decoder_inputs, target_weights, _, _ = model.get_batch({bucket_id: [(token_ids, [1])]},
                                                         bucket_id, model.batch_size, type=0)

            # print("bucket_id: ", bucket_id)
            # print("encoder_inputs:", encoder_inputs)
            # print("decoder_inputs:", decoder_inputs)
            # print("target_weights:", target_weights)

            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

            # print("output_logits", np.shape(output_logits))

            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # print(outputs)
            if util.EOS_ID in outputs:
                outputs = outputs[:outputs.index(util.EOS_ID)]

            print(" ".join([tf.compat.as_str_any(rev_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

if __name__ == '__main__':
    test(config.gen_config)
