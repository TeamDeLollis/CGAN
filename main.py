import argparse
import tensorflow as tf
from ops import *
from utils import *
from cyclegan import *

tf.compat.v1.set_random_seed(2)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', type=str, default='train')
args = parser.parse_args()


def main(_):
    tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        model = Cyclegan(sess)
        if args.mode == 'train':
            model.train()
        else:
            model.test()


if __name__ == '__main__':
    tf.compat.v1.app.run()
