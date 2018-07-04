import numpy as np
import tensorflow as tf
import sys
sys.path.append("../readers")
from reader_ptb import Reader_PTB
from reader_news20 import Reader_News20
from sklearn.datasets import fetch_20newsgroups

import pickle

import time

class LM():
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, session, reader, topic_model,
                 learning_rate=1.0,
                 embedding_size=200,
                 hidden_size = 200,
                 max_gradients=5,
                 num_layers=2,
                 keep_prob=1,
                 lr_decay = 0.5):

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
        # initialize model

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def data_type(self):
        return tf.float32

    def create_model(self):
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, None])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, None])
        self.keep_prob = tf.placeholder(tf.float32)



        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
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
        softmax_w = tf.get_variable(
            "softmax_w", [self.hidden, self.vocab_size], dtype=self.data_type())
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=self.data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
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

        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        self.init_op = tf.global_variables_initializer()
        self.all_saver = tf.train.Saver()
        self.sess.run(self.init_op)

        #INFERENCE
        self.initial_state_inf = cell.zero_state(1, self.data_type())
        self.input_inf = tf.placeholder(tf.int32,[1,1])
        inputs_inf = tf.nn.embedding_lookup(embedding, self.input_inf)
        #rnn step
        (cell_output_inf, self.state_inf) = cell(inputs_inf[:, 0, :], self.initial_state_inf)
        output_inf = [cell_output_inf]
        output_inf = tf.reshape(tf.concat(axis=1, values=output_inf), [-1, self.hidden])
        logits_inf = tf.matmul(output_inf, softmax_w) + softmax_b
        self.sample = tf.multinomial(logits_inf, 1)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


    def do_sample(self, seed, num_samples):
        """Sampled from the model"""
        samples = []
        state = self.sess.run(self.initial_state_inf)
        fetches = [self.state_inf, self.sample]
        sample = None
        if seed != "":
            for x in seed:
                feed_dict = {}
                feed_dict[self.keep_prob] = 1.0
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


    def run_epoch(self, data, is_train=False, verbose=False):
        """Runs the model on the given data."""
        epoch_size = sum( [len(x["doc_lm"]) for x in data])
        print("epoch size", epoch_size)
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
            for i, (x, y) in enumerate(doc["doc_lm"]):
                # print("x" , x)
                if x == []:
                    misses =+ 1
                    print( "whaay")
                    continue
                feed_dict = {}
                feed_dict[self.keep_prob] =keep_prob
                feed_dict[self.input_data] = x
                feed_dict[self.targets] = y
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

                if steps%100 == 0: #verbose and steps % (epoch_size // 10) == 10:
                    print("%.3f perplexity: %.3f speed: %.0f wps" %
                          (steps * 1.0 / epoch_size, np.exp(costs / iters),
                           iters * self.batch_size / (time.time() - start_time)))

                if steps%1000 == 0:
                    seed_for_sample = ["<EOS>"]
                    print("Seed: %s" % self.pretty_print([self.reader.lm_word2id[x] for x in seed_for_sample], self.reader.lm_id2word))
                    print("Sample: %s" % self.pretty_print(self.do_sample([self.reader.lm_word2id[word] for word in seed_for_sample],
                                                                          max(5 * (len(seed_for_sample) + 1), 100)), self.reader.lm_id2word))



        print("total steps:", steps)
        print("misses:", misses)

        return np.exp(costs / iters)

    def pretty_print(self, items, id2word):
        return ' '.join([id2word[x] for x in items])


    def train_model(self, epochs=20, save_path_lm_model="language_model"):

        seed_for_sample = ["the"]


        print("Seed: %s" % self.pretty_print([self.reader.lm_word2id[x] for x in seed_for_sample], self.reader.lm_id2word))
        print("Sample: %s" % self.pretty_print(self.do_sample([self.reader.lm_word2id[word] for word in seed_for_sample],
                                                max(5 * (len(seed_for_sample) + 1), 10)), self.reader.lm_id2word))

        for epoch in range(epochs):
            lr_decay = self.lr_decay ** max(epoch - epochs, 0.0)
            self.assign_lr(self.sess, self.learning_rate * lr_decay)
            train_ppl = self.run_epoch(data=self.reader.train, is_train=True, verbose=True)
            valid_ppl = self.run_epoch(data=self.reader.valid)
            print("epoch", epoch)
            print("train ppl", train_ppl)
            print("valid ppl:", valid_ppl)
            print("Seed: %s" % self.pretty_print([self.reader.lm_word2id[x] for x in seed_for_sample], self.reader.lm_id2word))
            print("Sample: %s" % self.pretty_print(self.do_sample([self.reader.lm_word2id[word] for word in seed_for_sample],
                                                          max(5 * (len(seed_for_sample) + 1), 10)), self.reader.lm_id2word))





if __name__ == '__main__':
    # reader = Reader_PTB(datapath="../data/PTB", length_batch= 20, batch_size=20)
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
    pickle.dump(dataset, open("../data/news20/news20.p" , "wb"))
    dataset = pickle.load(open("../data/news20/news20.p" , "rb"))
    reader = Reader_News20(dataset)
    pickle.dump(reader, open("../saved/news20reader.p", "wb"))
    reader = pickle.load(open("../saved/news20reader.p", "rb"))

    with tf.Session() as sess:
        m = LM(session=sess,
               reader=reader,
               topic_model="temp/language_model_simple.ckpt")
        print("vocab size:",  reader.lm_vocab_size)

        # m.initalize_from_trained_model(saved_model_path="temp/language_model_simple.ckpt" )
        # m.generate_sentence()
        m.train_model(epochs=60,save_path_lm_model="temp/language_model_simple.ckpt" )





