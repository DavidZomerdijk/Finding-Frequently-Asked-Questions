import numpy as np
import tensorflow as tf
import sys
sys.path.append("../readers")
sys.path.append("../tm")
from reader_ptb import Reader_PTB
from reader_news20 import Reader_News20
from reader_apnews import Reader_APNEWS
from RNN import tmGRUCell, tmLSTMCell
from LDA import LDA
import pickle

import time

class LOG():
    def __init__(self, file_path):
        self.filepath = file_path

    def log(self, string, verbose=True):
        with open(self.filepath , "a") as f:
            f.write(str(string)+ "\n" )
        if verbose:
            print(string)

class LM():
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, session, reader, topic_model, logger_tm_prog, logger_results, savepath_best,
                 learning_rate=1.0,
                 embedding_size=650,
                 hidden_size = 650,
                 max_gradients=5,
                 num_layers=1,
                 keep_prob=0.5,
                 lr_decay = 0.8,):

        self.embedding_size = embedding_size
        self.num_steps= reader.length_batch
        self.batch_size = reader.batch_size
        self.hidden = hidden_size
        self.lr_decay = lr_decay
        self.tm = topic_model
        self.sess = session
        self.reader = reader
        self.vocab_size = self.reader.lm_vocab_size
        self.learning_rate =  learning_rate
        self.name_model = "language model"
        self.num_layers = num_layers
        self.keep_prob_during_training = keep_prob
        self.max_grad_norm = max_gradients
        self.create_model()
        self.logger = logger_tm_prog
        self.logger_results = logger_results
        self.savepath_best = savepath_best
        # initialize model

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def data_type(self):
        return tf.float32

    def create_model(self):
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, None])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.topic = tf.placeholder(dtype=tf.float32, shape=[self.tm.n_topics])



        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.embedding_size, forget_bias=0.0, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)

        self.initial_state = cell.zero_state(self.batch_size, self.data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=self.data_type())
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        inputs = tf.nn.dropout(inputs, self.keep_prob)

        outputs = []
        state = self.initial_state


        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.hidden])

        print("output shape", output.shape)
        output_1d = output.shape[0]
        print(" output_1d", output_1d)
        tiled_topics = tf.tile(self.topic, [output_1d])
        print("tiled topics", tiled_topics.shape)
        topics = tf.reshape(tiled_topics , [output_1d , self.tm.n_topics])
        print(" topics", topics.shape)
        output_concatenated = tf.concat([output, topics], axis=1)

        softmax_w = tf.get_variable(
            "softmax_w", [self.hidden + self.tm.n_topics, self.vocab_size], dtype=self.data_type())
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=self.data_type())



        logits = tf.matmul(output_concatenated, softmax_w) + softmax_b
        # self.sample = tf.multinomial(logits, 1)

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([self.batch_size * self.num_steps], dtype=self.data_type())])

        self.cost = tf.reduce_sum(loss) / self.batch_size
        self.final_state = state

        self._lr = tf.Variable(0., trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          self.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # optimizer = tf.train.AdamOptimizer(0.001)

        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        self.init_op = tf.global_variables_initializer()
        self.all_saver = tf.train.Saver()
        self.sess.run( self.init_op )

        #INFERENCE
        self.initial_state_inf = cell.zero_state(1, self.data_type())
        self.input_inf = tf.placeholder(tf.int32,[1,1])
        inputs_inf = tf.nn.embedding_lookup(embedding, self.input_inf)
        #rnn step
        (cell_output_inf, self.state_inf) = cell(inputs_inf[:, 0, :], self.initial_state_inf)
        output_inf = [cell_output_inf]
        output_inf = tf.reshape(tf.concat(axis=1, values=output_inf), [-1, self.hidden])
        print("output inf", output_inf.shape)
        topic_reshaped = tf.reshape(self.topic, (1, self.tm.n_topics))
        print("topics2", topic_reshaped)
        output_inf_concat = tf.concat([output_inf, topic_reshaped], axis=1)
        print("output inf", output_inf_concat.shape)
        logits_inf = tf.matmul(output_inf_concat, softmax_w) + softmax_b
        self.sample = tf.multinomial(logits_inf, 1)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


    def do_sample(self, seed, num_samples, topic):
        """Sampled from the model"""
        samples = []
        state = self.sess.run(self.initial_state_inf)
        fetches = [self.state_inf, self.sample]
        sample = None
        if seed != "":
            for x in seed:
                feed_dict = {}
                feed_dict[self.keep_prob] = 1.0
                feed_dict[self.topic] = topic
                feed_dict[self.input_inf] = [[x]]
                for layer_num, (c, h) in enumerate(self.initial_state_inf):
                    feed_dict[c] = state[layer_num].c
                    feed_dict[h] = state[layer_num].h

                state, sample = self.sess.run(fetches, feed_dict)
        if sample is not None:
            samples.append(sample[0][0])
        else:
            samples.append(0)
        k = 1
        while k < num_samples:
            feed_dict = {}
            feed_dict[self.keep_prob] = 1.0
            feed_dict[self.topic] = topic
            feed_dict[self.input_inf] = [[samples[-1]]]
            for layer_num, (c, h) in enumerate(self.initial_state_inf):
                feed_dict[c] = state[layer_num].c
                feed_dict[h] = state[layer_num].h
            state, sample = self.sess.run(fetches, feed_dict)
            samples.append(sample[0][0])
            k += 1
        return samples


    def initalize_from_trained_model(self, saved_model_path):
        self.all_saver.restore(self.sess, saved_model_path)

    def print_topic_sample(self, seed=["<EOS>"], sample_length=20):

        print("Seed: %s" % self.pretty_print([self.reader.lm_word2id[x] for x in seed], self.reader.lm_id2word))

        for i in range(self.tm.n_topics):
            topic = np.zeros((self.tm.n_topics))
            topic[i] = 1
            self.logger.log("topic %i: %s" % (i ,self.pretty_print(self.do_sample(seed= [self.reader.lm_word2id[word] for word in seed],
                                                                                  num_samples =max(5 * (len(seed) + 1), sample_length),
                                                                                  topic=topic),
                                                                   self.reader.lm_id2word)))

    def run_epoch(self, data, is_train=False, verbose=True):
        """Runs the model on the given data."""
        epoch_size = sum( [len(x["doc_lm"]) for x in data])
        print("epoch_size", epoch_size)

        start_time = time.time()
        costs = 0.0
        iters = 0
        state = self.sess.run(self.initial_state)

        if is_train:
            keep_prob = self.keep_prob_during_training
            fetches = [self.cost, self.final_state, self.train_op]
        else:
            keep_prob = 1.0
            fetches = [self.cost, self.final_state]

        steps = 0
        misses = 0
        for doc in data:
            # topics = np.array( [self.tm.get_topic_distribution( [np.array(doc["doc_tm"])])[0] for _ in range(self.batch_size)])# self.tm.get_topic_distribution( [np.array(doc["doc_tm"])])# for _ in range(len(doc["doc_lm"]))])
            topics = self.tm.get_topic_distribution( [np.array(doc["doc_tm"])])[0]

            for i, (x, y) in enumerate(doc["doc_lm"]):
                feed_dict = {}
                feed_dict[self.keep_prob] = keep_prob
                feed_dict[self.input_data] = x
                feed_dict[self.targets] = y
                feed_dict[self.topic] = topics # np.array([topics[0]])
                for layer_num, (c, h) in enumerate(self.initial_state):
                    feed_dict[c] = state[layer_num].c
                    feed_dict[h] = state[layer_num].h

                if is_train:
                    cost, state, _ = self.sess.run(fetches, feed_dict)
                else:
                    cost, state = self.sess.run(fetches, feed_dict)

                costs += cost
                iters += self.num_steps
                steps += 1

                # self.all_saver.save(self.sess, "temp/language_model.ckpt")

                if steps%10000 == 0 and verbose:#verbose and steps % (epoch_size // 10) == 10:
                    self.logger.log("%.3f perplexity: %.3f speed: %.0f wps" %
                                    (steps * 1.0 / epoch_size, np.exp(costs / iters),
                                     iters * self.batch_size / (time.time() - start_time)))


        if  verbose:
            self.print_topic_sample(sample_length=30)



        print("total steps:", steps)
        print("misses:", misses)

        return np.exp(costs / iters)

    def pretty_print(self, items, id2word):
        return ' '.join([id2word[x] for x in items])


    def train_model(self, epochs=20, save_path_lm_model="tmp/language_model.ckpt"):
        best_valid_ppl = 999999999
        self.print_topic_sample()
        for epoch in range(epochs):
            lr_decay = self.lr_decay ** max(epoch - epochs, 0.0)
            self.assign_lr(self.sess, self.learning_rate * lr_decay)
            train_ppl = self.run_epoch(data=self.reader.train, is_train=True, verbose=True)
            valid_ppl = self.run_epoch(data=self.reader.valid)
            test_ppl = self.run_epoch(data=self.reader.test)
            self.logger.log("epoch", epoch)
            self.logger.log("train ppl:" + str(train_ppl))
            self.logger.log("valid ppl:" + str(valid_ppl))

            self.logger_results.log(str(epoch) + " " + str(train_ppl)+ " " + str(valid_ppl) + " " + str(test_ppl))


            # log results
            if best_valid_ppl > valid_ppl:
                self.all_saver.save(self.sess, save_path_lm_model)
                best_valid_ppl = valid_ppl
                with open(self.savepath_best, "w") as f:
                    f.write(str(epoch) + " " + str(train_ppl) + " " + str(valid_ppl) + " " + str(test_ppl) + "\n")

                with open(self.savepath_best, "a") as f:
                    for i in range(self.tm.n_topics):
                        topic = np.zeros(( self.tm.n_topics))
                        topic[i] = 1
                        f.write("topic " + str(i) + " " + str(self.pretty_print(self.do_sample(seed= [self.reader.lm_word2id[word] for word in ["<EOS>"]],
                                                                                               num_samples =max(5 * (len(["<EOS>"]) + 1), 30),
                                                                                               topic=topic),
                                                                                self.reader.lm_id2word)) + "\n")




if __name__ == '__main__':

    # bs = 5
    # bl = 5
    # reader = Reader_PTB(datapath="../data/PTB", length_batch= 20, batch_size=20)
    # print("start preprocessing news20")
    # dataset_news20 = pickle.load(open("../data/news20/news20.p" , "rb"))
    # reader_news20 = Reader_News20(data=dataset_news20, n_features=1000,
    #                               lm_minimum_freq=5, train_perc=0.6, valid_perc= 0.2, language="english"
    #                               , length_batch=bl, batch_size=bs)
    # pickle.dump(reader_news20, open("../saved/news_20.p", "wb"))
    #
    # print("start preprocessing APnews")
    # reader_apnews = Reader_APNEWS(datapath="../data/apnews/apnews.dat", n_features=1000,
    #                               lm_minimum_freq=5, train_perc=0.6, valid_perc= 0.2, language="english"
    #                               , length_batch=bl, batch_size=bs, sample_size=10000)
    # pickle.dump(reader_apnews, open("../saved/apnews.p", "wb"))


    reader_apnews = pickle.load(open("../saved/apnews.p", "rb"))
    # reader_news20 = pickle.load(open("../saved/news_20.p", "rb"))

    with tf.Session() as sess:
        print("start training tm apnews")
        # tm = LDA(reader_apnews , topics=20, max_iter=50 )
        # tm.experiments( save_location="results/AN_tm_topics.txt")
        # pickle.dump(tm, open("tm.p", "wb"))

        tm = pickle.load(open("tm.p", "rb"))
        tm.experiments( save_location="results/AN_2_tm_topics_med_1_keep.txt")
        print("start training lm")
        m = LM(session=sess,
               reader=reader_apnews,
               topic_model=tm,
               num_layers=2,
               logger_tm_prog= LOG(file_path="results/AN_2_lm_topic_progression_med_1_keep.txt"),
               logger_results = LOG(file_path="results/AN_2_results_med_1_keep"),
               savepath_best = "results/AN_2_best_topic_med_300_1_keep.txt",
               embedding_size=650,
               hidden_size = 650,
               max_gradients=5,
               keep_prob=1.0,
               lr_decay=0.8)

        print("vocab size:",  reader_apnews.lm_vocab_size)

        # m.initalize_from_trained_model(saved_model_path="tmp/language_model.ckpt" )
        # m.generate_sentence()
        m.train_model(epochs=40,save_path_lm_model="temp/language_model.ckpt" )



        # # news20
        # tf.reset_default_graph()
        #
        # with tf.Session() as sess:
        #     print("start training tm")
        #     tm = LDA(reader_news20 , topics=20 )
        #     tm.experiments( save_location="results/N20_tm_topics.txt")
        #     # pickle.dump(tm, open("tm.p", "wb"))
        #
        #     # tm = pickle.load(open("tm.p", "rb"))
        #     print("start training lm")
        #     m = LM(session=sess,
        #            reader=reader_news20,
        #            topic_model=tm,
        #            num_layers=2,
        #            logger_tm_prog= LOG(file_path="results/N20_lm_topic_progression.txt"),
        #            logger_results = LOG(file_path="results/N20_results"),
        #            savepath_best = "results/N20_best_topic.txt",
        #            embedding_size=650,
        #            hidden_size = 650,
        #            max_gradients=5,
        #            keep_prob=0.5,
        #            lr_decay=0.8)
        #
        #     print("vocab size:",  reader_news20.lm_vocab_size)
        #
        #     # m.initalize_from_trained_model(saved_model_path="temp/language_model_simple.ckpt" )
        #     # m.generate_sentence()
        #     m.train_model(epochs=40,save_path_lm_model="temp/language_model_simple.ckpt" )





