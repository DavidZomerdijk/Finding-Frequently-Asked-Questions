import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append("../readers")
sys.path.append("../")
from utils import pp
# from ntm_batch import NTM
from nvdm import NVDM
# from reader import TextReader
import pickle as p

from reader_news20 import Reader_News20
from reader_apnews import Reader_APNEWS

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_integer("max_iter", 1500000, "Maximum of iteration [450000]")
flags.DEFINE_integer("h_dim", 50, "The dimension of latent variable [50]")
flags.DEFINE_integer("embed_dim", 256, "The dimension of word embeddings [500]")
flags.DEFINE_string("dataset", "apnews", "The name of dataset [ptb, news_20 apnews]")
flags.DEFINE_string("model", "ntm", "The name of model [nvdm, nasm]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_boolean("forward_only", False, "False for training, True for testing [False]")
flags.DEFINE_boolean("perform_experiments", True, "Either do the experiments at the end or not")
FLAGS = flags.FLAGS


MODELS = {
    # 'ntm': NTM,
    'nvdm': NVDM
}


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if  FLAGS.dataset=='ptb':
        print("Nothing to see here")
        # data_path = "../data/%s" % FLAGS.dataset
        # reader = TextReader(data_path)
    elif FLAGS.dataset=="news_20":
        # dataset = p.load(open("../data/news20/news20.p" , "rb"))
        # reader = Reader_News20(dataset, n_features=1000)
        # p.dump(reader, open("../saved/news20reader.p", "wb"))
        reader = p.load(open("../saved/news20reader.p", "rb"))
    elif FLAGS.dataset=="apnews":
        # reader = Reader_APNEWS(datapath="../data/apnews/apnews.dat", n_features=1000, sample_size=10000)
        # pickle.dump(reader, open("../saved/apnews.p", "wb"))
        reader = p.load(open("../saved/apnews.p", "rb"))

    with tf.Session() as sess:
        m = MODELS['nvdm']
        model = m(sess, reader, FLAGS.dataset,
                  embed_dim=FLAGS.embed_dim, h_dim=FLAGS.h_dim,
                  learning_rate=FLAGS.learning_rate, max_iter=FLAGS.max_iter,
                  checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.forward_only:
            model.load(FLAGS.checkpoint_dir)
        else:
            print("start training")
            model.train()

        if FLAGS.perform_experiments:
            model.experiments()



if __name__ == '__main__':
    tf.app.run()