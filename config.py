import os

# configuration options for discriminator network
class disc_config(object):
    batch_size = 256
    lr = 0.2
    lr_decay = 0.9
    vocab_size = 35000
    embed_dim = 512
    steps_per_checkpoint = 100
    num_layers = 2
    train_dir = './dis_data/'
    name_model = "disc_model"
    name_loss = "disc_loss"
    max_len = 50
    piece_size = batch_size * steps_per_checkpoint
    piece_dir = "./dis_data/batch_piece/"
    valid_num = 100
    init_scale = 0.1
    num_class = 2
    keep_prob = 0.5
    max_grad_norm = 5
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    dis_pre_train_step = 80000


# configuration options for generator network
class gen_config(object):
    beam_size = 7
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 128
    emb_dim = 512
    num_layers = 2
    vocab_size = 35000
    train_dir = "./gen_data/"
    name_model = "st_model"
    name_loss = "gen_loss"
    teacher_loss = "teacher_loss"
    reward_name = "reward"
    max_train_data_size = 0
    steps_per_checkpoint = 100
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]
    gen_pre_train_step = 400000