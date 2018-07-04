# Using this code one can try to run the model with different parameters

import numpy as np
import tensorflow as tf
import sys
sys.path.append("../readers")
from reader_ptb import Reader_PTB
from reader_apnews import Reader_APNEWS
from lm_standard import LM
from lm_standard_apnews import LM_apnews

if True:
    reader = Reader_PTB(datapath="../data/PTB", length_batch= 35, batch_size=20)

    for hidden in [100,200,300,400,500,600,650,700,800,900,1000]:
        for layers in [1,2,3]:
            with tf.Session() as sess:
                m = LM(session=sess,
                       learning_rate=1.0,
                       embedding_size=hidden,
                       hidden_size = hidden,
                       max_gradients=5,
                       num_layers=layers,
                       lr_decay = 0.8,
                       keep_prob=0.5,
                       reader= reader,
                       topic_model="temp/language_model_simple.ckpt")

                m.train_model(epochs=50, save_path_lm_model="temp/language_model_simple.ckpt" )
                m.save_results(save_path= "experiment_results.txt")
            tf.reset_default_graph()

if False:
    reader = Reader_APNEWS(datapath="../data/apnews/apnews.dat"  , length_batch=10, batch_size=5)
    for layers in [1,2,3]:
        for hidden in [200,400,600,800,1000, 1200]:
            with tf.Session() as sess:
                m = LM_apnews(session=sess,
                       embedding_size=hidden,
                       hidden_size = hidden,
                       max_gradients=5,
                       num_layers=layers,
                       reader=reader,
                       topic_model="temp/language_model_simple.ckpt")

                m.train_model(epochs=20,save_path_lm_model="temp/language_model_simple.ckpt" )
                m.save_results(save_path= "experiment_results_apnews.txt")



