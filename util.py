import os
import re
from six.moves import urllib
from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

def basic_tokenizer(sentence):
  words = []
  for space_separated_fragment in sentence.strip().split():
    if type(space_separated_fragment) == bytes:
      space_separated_fragment = space_separated_fragment.decode()
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]

def create_vocabulary(vocabulary_path, data_path_list, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from disc_data %s" % (vocabulary_path, data_path_list))
    vocab = {}
    for data_path in data_path_list:
        with gfile.GFile(data_path, mode="r") as f:
          counter = 0
          for line in f:
            counter += 1
            if counter % 100000 == 0:
              print("  processing line %d" % counter)
            line = tf.compat.as_str_any(line)
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for w in tokens:
              word = _DIGIT_RE.sub("0", w) if normalize_digits else w
              if word in vocab:
                vocab[word] += 1
              else:
                vocab[word] = 1

    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
      for w in vocab_list:
        vocab_file.write(w + "\n")

def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary,
                      tokenizer=None, normalize_digits=True):
  if not gfile.Exists(target_path):
    print("Tokenizing disc_data in %s" % data_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocabulary, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_chitchat_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
  train_path = os.path.join(data_dir, "chitchat.train")
  dev_path = os.path.join(data_dir, "chitchat.dev")

  answer_train_ids_path = train_path + (".ids%d.answer" % vocabulary_size)
  query_train_ids_path = train_path + (".ids%d.query" % vocabulary_size)
  data_to_token_ids(train_path + ".answer", answer_train_ids_path, vocabulary, tokenizer)
  data_to_token_ids(train_path + ".query", query_train_ids_path, vocabulary, tokenizer)

  # Create token ids for the development disc_data.
  answer_dev_ids_path = dev_path + (".ids%d.answer" % vocabulary_size)
  query_dev_ids_path = dev_path + (".ids%d.query" % vocabulary_size)
  data_to_token_ids(dev_path + ".answer", answer_dev_ids_path, vocabulary, tokenizer)
  data_to_token_ids(dev_path + ".query", query_dev_ids_path, vocabulary, tokenizer)

  return (query_train_ids_path, answer_train_ids_path,
          query_dev_ids_path, answer_dev_ids_path)

def hier_prepare_disc_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
  train_path = os.path.join(data_dir, "train")
  dev_path = os.path.join(data_dir, "dev")

  # Create token ids for the training disc_data.
  query_train_ids_path = train_path + (".ids%d.query" % vocabulary_size)
  answer_train_ids_path = train_path + (".ids%d.answer" % vocabulary_size)
  gen_train_ids_path = train_path + (".ids%d.gen" % vocabulary_size)

  data_to_token_ids(train_path + ".query", query_train_ids_path, vocabulary, tokenizer)
  data_to_token_ids(train_path + ".answer", answer_train_ids_path, vocabulary, tokenizer)
  data_to_token_ids(train_path + ".gen", gen_train_ids_path, vocabulary, tokenizer)

  # Create token ids for the development disc_data.
  query_dev_ids_path = dev_path + (".ids%d.query" % vocabulary_size)
  answer_dev_ids_path = dev_path + (".ids%d.answer" % vocabulary_size)
  gen_dev_ids_path = dev_path + (".ids%d.gen" % vocabulary_size)

  data_to_token_ids(dev_path + ".query", query_dev_ids_path, vocabulary, tokenizer)
  data_to_token_ids(dev_path + ".answer", answer_dev_ids_path, vocabulary, tokenizer)
  data_to_token_ids(dev_path + ".gen", gen_dev_ids_path, vocabulary, tokenizer)

  return (query_train_ids_path, answer_train_ids_path, gen_train_ids_path,
          query_dev_ids_path, answer_dev_ids_path, gen_dev_ids_path)
