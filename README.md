# Adversarial Learning for Neural Dialogue Generation

the paper: Adversarial Learning for Neural Dialogue Generation https://arxiv.org/pdf/1701.06547.pdf

将数据集中的chitchat.train.query、chitchat.train.answer、chitchat.dev.query和chitchat.dev.answer放入gen_data文件夹下即可训练，dis_data初始为空，经过下面步骤2会自动生成。

训练步骤：

1.python gen_pre_train.py 预训练生成器。需要有gen_data/chitchat.dev.answer等四个文件才能运行，也可在config.py中修改各种设置。运行后，gen_data/checkpoints中将储存经过预训练的权值文件。

2.python gen_data.py 读取生成器预训练后的权值，为判别器预训练过程生成数据。运行后，将会生成dis_data/dev.answer等六个文件。

3.python dis_pre_train.py 预训练判别器。需要有上一步生成的六个文件才能运行。运行后，dis_data/checkpoints中将储存经过预训练的权值文件。

4.python train.py读取经过预训练的生成器和判别器权值，进行对抗训练，并将权值文件保存。

5.测试程序为test.py。执行python test.py后，程序将读取gen_data/checkpoints中训练好的权值文件，进行人机交互——程序将等待用户输入，然后根据用户输入，输出回应，直到用户输入Ctrl+Z退出。