# implementation is based at https://github.com/akashgit/autoencoding_vi_for_topic_models/blob/master/models/prodlda.py

import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import pickle as p
import copy
sys.path.append("../readers")
sys.path.append("../")
sys.path.append("../misc")
sys.path.append("../../data")
sys.path.append("../data")
from reader_prodlda_lstm import Reader_lstm_shortened
from tensorflow.contrib.rnn import  MultiRNNCell, ResidualWrapper, DropoutWrapper
# from reader_news20 import Reader_News20
# from news20 import News20
from base import Model
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)
def log_dir_init(fan_in, fan_out,topics=50):
    return tf.log((1.0/topics)*tf.ones([fan_in, fan_out]))

tf.reset_default_graph()
class VAE(Model):
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """


    def __init__(self, session, reader, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.002, batch_size=100, max_iter=800000):
        self.max_iter = max_iter
        self.inference = tf.placeholder(tf.bool)
        self.sess= session
        self.reader = reader
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.name_model = "prodla"
        self.embedding_size = 100
        self.lstm_num_layers = 1
        self.lstm_num_hidden = 100
        self.vocab_size = self.reader.lm_vocab_size
        self.weight_initializer = tf.contrib.layers.xavier_initializer()

        print('Learning Rate:', self.learning_rate)

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        self.keep_prob = tf.placeholder(tf.float32)

        # self.h_dim = float(network_architecture["n_z"])
        self.h_dim = network_architecture["n_z"]
        self.a = 1*np.ones((1 , self.h_dim)).astype(np.float32)

        self.mu2 = tf.constant((np.log(self.a).T-np.mean(np.log(self.a),1)).T)

        self.var2 = tf.constant(  ( ( (1.0/self.a)*( 1 - (2.0/self.h_dim) ) ).T +
                                    ( 1.0/(self.h_dim*self.h_dim) )*np.sum(1.0/self.a,1) ).T  )

        self._create_network()
        self._create_loss_optimizer()

        init =  tf.global_variables_initializer() #tf.initialize_all_variables()

        self.sess.run(init)



    # I got inspiration from: https://github.com/Chung-I/Variational-Recurrent-Autoencoder-Tensorflow/blob/master/seq2seq_model.py
    def encoder_to_latent(self,
                          encoder_state,
                          embedding_size,
                          latent_dim,
                          num_layers,
                          parameters=1,
                          dtype=None):

        concat_state_size = num_layers * embedding_size
        concat_state_size *= 2
        encoder_state = list(map(lambda state_tuple: tf.concat( state_tuple, 1), encoder_state))
        encoder_state = tf.concat( encoder_state, 1)

        with tf.variable_scope('encoder_to_latent'):
            w = tf.get_variable("w",[concat_state_size, parameters * latent_dim], initializer=self.weight_initializer,
                                dtype=dtype)
            b = tf.get_variable("b", [parameters * latent_dim], initializer=tf.constant_initializer(0.0), dtype=dtype)

            # mean_logvar = tf.matmul(encoder_state, w) + b
            # mean, logvar = tf.split(1, 2, mean_logvar)

        return tf.matmul(encoder_state, w) + b



    def _create_network(self):
        self.network_weights = self._initialize_weights(**self.network_architecture)
        self.z_mean,self.z_log_sigma_sq = \
            self._recognition_network(self.network_weights["weights_recog"],
                                      self.network_weights["biases_recog"])

        n_z = self.network_architecture["n_z"]
        # eps = tf.random_normal((self.batch_size, n_z), 0, 1,
        #                        dtype=tf.float32)
        eps = tf.random_normal((1, self.h_dim), 0, 1, dtype=tf.float32)
        self.sigma = tf.exp(self.z_log_sigma_sq)

        #inference step
        self.z = tf.cond(self.inference, lambda:  self.z_mean ,lambda: tf.add(self.z_mean, tf.multiply(tf.sqrt( self.sigma ), eps)))



        self.x_reconstr_mean = \
            self._generator_network(self.z,self.network_weights["weights_gener"])



    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.get_variable('h1',[n_input, n_hidden_recog_1]),
            'h2': tf.get_variable('h2',[n_hidden_recog_1, n_hidden_recog_2]),
            'out_mean': tf.get_variable('out_mean',[n_hidden_recog_2, n_z]),
            'out_log_sigma': tf.get_variable('out_log_sigma',[n_hidden_recog_2, n_z])}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h2': tf.Variable(xavier_init(n_z, n_hidden_gener_1))}

        return all_weights

    def _recognition_network(self, weights, biases):
        self.X_in = tf.placeholder(tf.int32, [None, None])
        self.X = tf.contrib.layers.embed_sequence(self.X_in, self.vocab_size, self.embedding_size)
        self.seq_lengths = tf.placeholder( dtype=tf.int32, shape=(None,), name='inputs_length_x')

        # encoder
        cell_fw = MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.lstm_num_hidden),
                                                                  output_keep_prob=self.keep_prob) for _ in range(self.lstm_num_layers)])

        cell_bw = MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.lstm_num_hidden),
                                                                  output_keep_prob=self.keep_prob) for _ in range(self.lstm_num_layers)])


        output, hidden = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = cell_fw, cell_bw=cell_bw, inputs=self.X , sequence_length=(self.seq_lengths) ,
            dtype=tf.float32 ,  time_major=False)

        forward_out = output[0]
        backward_out = output[1]

        print("forward", forward_out)
        print("backward",backward_out)

        seq_mask = tf.sequence_mask( self.seq_lengths, dtype=tf.float32)

        # stack sequence mask to mask size of output
        seq_mask_stacked = tf.stack( [ seq_mask for _ in range(self.lstm_num_hidden)] ,axis=-1)
        # stack sequence lengths to mask size of output
        seq_length_stacked = tf.stack( [self.seq_lengths for _ in range(self.lstm_num_hidden)], axis=-1 )

        # masked output
        masked_fw = tf.multiply(forward_out, seq_mask_stacked)
        masked_bw = tf.multiply(backward_out, seq_mask_stacked)

        # get mean of masked output
        mean_lstm_output_fw = tf.reduce_sum(masked_fw, axis=1) / tf.cast(seq_length_stacked, tf.float32)
        mean_lstm_output_bw = tf.reduce_sum(masked_bw, axis=1) / tf.cast(seq_length_stacked, tf.float32)

        mean_lstm_output = tf.concat( [mean_lstm_output_bw, mean_lstm_output_fw], axis=1)

        print( "masked", mean_lstm_output_bw)
        print("length", seq_mask_stacked)
        print("length2", seq_length_stacked)
        print("seq length", self.seq_lengths)
        print("seq length2",  tf.squeeze(self.seq_lengths, axis=0))
        print("seq mask", seq_mask)
        # print("output", self.encoder_outputs)


        # self.z_fw =  self.encoder_to_latent( self.encoder_state,
        #                                      self.lstm_num_hidden,
        #                                      100,
        #                                      num_layers= 1)#.lstm_num_layers)
        # Generate probabilistic encoder (recognition network)
        # layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
        #                                    biases['b1']))
        # layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
        #                                    biases['b2']))
        # layer_do = tf.nn.(layer_2, self.keep_prob)
        # layer_do = tf.nn.dropout(self.z_fw, self.keep_prob)

        layer_do = tf.nn.dropout(mean_lstm_output, self.keep_prob)
        z_mean = tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out_mean']),
                                                     biases['out_mean']))
        z_log_sigma_sq = \
            tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out_log_sigma']),
                                                biases['out_log_sigma']))


        return (z_mean, z_log_sigma_sq)

    def _generator_network(self,z, weights):
        self.t = tf.nn.softmax(z)
        self.layer_do_0 = tf.nn.dropout(self.t, self.keep_prob)
        x_reconstr_mean = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.add(
            tf.matmul(self.layer_do_0, weights['h2']),0.0)))
        return x_reconstr_mean


    def _create_loss_optimizer(self):

        self.x_reconstr_mean+=1e-10

        self.reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean),1)#/tf.reduce_sum(self.x,1)

        self.latent_loss = 0.5*( tf.reduce_sum(tf.div(self.sigma,self.var2),1)+
                                 tf.reduce_sum( tf.multiply(tf.div((self.mu2 - self.z_mean),self.var2),
                                                            (self.mu2 - self.z_mean)),1) - self.h_dim +
                                 tf.reduce_sum(tf.log(self.var2),1)  - tf.reduce_sum(self.z_log_sigma_sq  ,1) )

        self.cost = tf.reduce_mean(self.reconstr_loss) + tf.reduce_mean(self.latent_loss) # average over batch


        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.99).minimize(self.cost)

    def get_batch(self, data, batch_size):
        d = [ copy.deepcopy(data[ np.random.randint(len(data)-1, size=1)[0]]) for _ in range(batch_size)]
        X = np.array([x["doc_tm"] for x in d])
        sparse = [x["doc_tm_sparse"] for x in d]

        lstm_input_temp = [x["lstm_model"] for x in d]
        seq_length = np.array([ len(x) for x in lstm_input_temp])
        max_length = np.max(seq_length)
        for s in lstm_input_temp:
            if len(s) < max_length:
                for i in range(max_length - len(s)):
                    s.append( self.reader.lm_word2id["<PAD>"] )
        return {"x": X, 'seq_length': seq_length, "lstm_input": np.array(lstm_input_temp), "doc_tm_sparse" : sparse }

    def train(self, evaluate_every=100):
        print("commence training!")
        best_perplexity = 99999999

        with open("results/prodlda.txt", "w") as f:
            f.write("step train_perp valid_perp test_perp test_recall total_loss e_loss rec_loss\n")

        train = self.reader.train
        valid = self.reader.valid
        test = self.reader.test


        #dropout 0.4 is standard
        for step in range(self.max_iter):
            input_data = self.get_batch(train, self.batch_size)

            # data = [ copy.deepcopy(train[ np.random.randint(len(train)-1, size=1)[0]]) for _ in range(self.batch_size)]
            # X = np.array([x["doc_tm"] for x in data])

            opt, cost,emb = self.sess.run((self.optimizer, self.cost, self.network_weights['weights_gener']['h2']),
                                          feed_dict={self.x: input_data["x"],
                                                      self.seq_lengths: input_data["seq_length"],
                                                      self.X_in: input_data['lstm_input'],
                                                      self.keep_prob: 0.6,
                                                      self.inference: False})

            if step % evaluate_every == 0 :
                # print("step:", step, "train_loss", cost)


                train_stats = self.evaluate_model(train, step, verbose=False)
                valid_stats = self.evaluate_model(valid, step, verbose=False)
                test_stats = self.evaluate_model(test, step, verbose=False)

                with open("results/prodlda.txt", "a") as f:
                    f.write( str(step) + " "
                             +  str(train_stats['perplexity']) + " "
                             + str(valid_stats['perplexity'])  + " "
                             + str(test_stats['perplexity'])  + " "
                             + str(test_stats['recall_3'] ) + " "
                             + str(train_stats["loss"]) + " "
                             + str(train_stats["loss_latent"]) + " "
                             + str(train_stats["loss_rec"]) + "\n" )

                perplexity_valid = valid_stats['perplexity']

                print("step:", step,"loss:", train_stats['loss'] , "train perp:", train_stats["perplexity"],
                      "valid_perp:", valid_stats["perplexity"],
                      "test_perp:" , test_stats['perplexity'])
                if perplexity_valid < best_perplexity:
                    self.show_topics(verbose=False, save=True)
                    best_perplexity = perplexity_valid

        print("ready training")

    def evaluate_model(self, test_set, step, size_test_set=100, n_recall=3, verbose=True):
        recall = []
        # test = np.array([x["doc_tm"] for x in test_set[:size_test_set]])
        input_data = self.get_batch(test_set, self.batch_size)

        loss, loss_rec, loss_latent, output = self.sess.run((self.cost, self.reconstr_loss, self.latent_loss ,self.x_reconstr_mean ),
                                                            feed_dict={ self.x: input_data["x"],
                                                                        self.seq_lengths: input_data["seq_length"],
                                                                        self.X_in: input_data['lstm_input'],
                                                                        self.keep_prob: 1,
                                                                        self.inference: True})

        # print("loss", loss_latent)
        loss_rec_mean = sum(loss_rec) / len(loss_rec)
        loss_latent_mean = sum(loss_latent) / len(loss_latent)
        for i in range( size_test_set ):
            recall.append( self.recall(input_data["x"][i], output[i], n=n_recall))

        perplexity = self.perplexity(input_data)
        if verbose:
            print("train step:", step, "test-loss:", loss, "recall_n=3", sum(recall) / len(recall) , "perplexity" , perplexity)
        return {"loss": loss, "loss_rec" : loss_rec_mean, "loss_latent" : loss_latent_mean, "perplexity": perplexity, 'recall_3': sum(recall) / len(recall) }

    def perplexity(self, input_data, n_samples=100, print_errors=False):

        # test = [x['doc_tm'] for x in testset[:n_samples]]
        # test_idx = [x['doc_tm_sparse'] for x in testset[:n_samples]]
        perplexities = []

        x_hat = self.x_reconstr_mean.eval(feed_dict={self.x: input_data['x'],  self.keep_prob: 1,
                                                     self.seq_lengths: input_data["seq_length"],
                                                     self.X_in: input_data['lstm_input'],
                                                     self.inference: True})

        for i in range(n_samples):
            probs =  np.log(np.take(x_hat[i], input_data['doc_tm_sparse'][i]))
            perplexities.append( sum(probs) / len(probs))


        total_perplexity =  np.exp(-sum(perplexities) / n_samples)
        if print_errors:
            print("prodla perplexity on test_set",total_perplexity, "based on", n_samples, "samples.")
        return total_perplexity

    def show_topics(self,verbose=False, save=True, top_n_words=10, n_topics=-1,):
        t = np.zeros( (self.network_architecture["n_z"], self.network_architecture["n_z"]), int)
        np.fill_diagonal(t, 1)
        x_hats = self.sess.run( self.x_reconstr_mean, feed_dict={self.t: t, self.keep_prob: 1})


        topics = []
        for i in range(self.network_architecture["n_z"]):
            topic = []
            a = np.argsort(x_hats[i,:])[::-1]
            for idx in a:
                topic.append({"word" : self.reader.idx2word[idx], "index" : idx, "prob": x_hats[i,idx]})
            topics.append(topic)

        if verbose:
            print("top", top_n_words, "words for", self.name_model)
            for i, topic in enumerate(topics):
                print("topic:", i , " ".join([ topic[i]["word"] for i in range(top_n_words)]))
            print("".join(["-" for i in range(50)]))

        if save:
            with open("results/prodlda_topics.txt", "w") as f:
                for i, topic in enumerate(topics):
                    f.write( " ".join([ topic[i]["word"] for i in range(top_n_words)]) + "\n")



if __name__ == '__main__':


    reader = Reader_lstm_shortened( datapath="../../data/data.p", n_features=1000, lm_minimum_freq=5, train_perc=0.6, valid_perc= 0.2,
                         language="dutch",
                         length_batch=5,
                         batch_size=5,
                         n_samples=50000,
                         only_use_sample=True,max_sentence_length = 30)

    network_architecture =  dict(n_hidden_recog_1=200, # 1st layer encoder neurons
                                 n_hidden_recog_2=200, # 2nd layer encoder neurons
                                 n_hidden_gener_1=reader.vocab_size, # 1st layer decoder neurons #data_tr.shape[1],
                                 n_input=reader.vocab_size, # MNIST data input (img shape: 28*28) # data_tr.shape[1],
                                 n_z=20)  # dimensionality of latent space


    with tf.Session() as sess:
        model = VAE(sess, reader, network_architecture)
        model.train()