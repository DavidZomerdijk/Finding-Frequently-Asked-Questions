# code is run in trainer.py

import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from base import Model
import copy

class NVDM(Model):
  """Neural Varational Document Model"""

  def __init__(self, sess, reader, dataset="ptb",
               decay_rate=0.96, decay_step=10000, embed_dim=500,
               h_dim=50, learning_rate=0.001, max_iter=450000,
               checkpoint_dir="checkpoint"):
    """Initialize Neural Varational Document Model.

    params:
      sess: TensorFlow Session object.
      reader: TextReader object for training and test.
      dataset: The name of dataset to use.
      h_dim: The dimension of document representations (h). [50, 200]
    """
    self.sess = sess


    self.reader = reader
    #h_dim is topic dimension
    self.h_dim = h_dim
    self.embed_dim = embed_dim

    self.max_iter = max_iter
    self.decay_rate = decay_rate
    self.decay_step = decay_step
    self.checkpoint_dir = checkpoint_dir
    self.step = tf.Variable(0, trainable=False)
    self.lr = tf.train.exponential_decay(
      learning_rate, self.step, 10000, decay_rate, staircase=True, name="lr")

    # _ = tf.scalar_summary("learning rate", self.lr)

    self.dataset = dataset
    self._attrs = ["h_dim", "embed_dim", "max_iter", "dataset",
                   "learning_rate", "decay_rate", "decay_step"]

    self.build_model()

  def build_model(self):
    # this is the document
    self.x = tf.placeholder(tf.float32, [self.reader.vocab_size], name="input")
    # this is
    self.x_idx = tf.placeholder(tf.int32, [None], name="x_idx")
    self.inference =  tf.placeholder(tf.bool)

    self.build_encoder()
    self.build_generator()

    # Kullback Leibler divergence
    self.e_loss = -0.5 * tf.reduce_sum(1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq))

    # Log likelihood
    self.g_loss = -tf.reduce_sum(tf.log(tf.gather(self.p_x_i, self.x_idx) + 1e-10))

    self.loss = self.e_loss + self.g_loss

    self.encoder_var_list, self.generator_var_list = [], []
    for var in tf.trainable_variables():
      if "encoder" in var.name:
        self.encoder_var_list.append(var)
      elif "generator" in var.name:
        self.generator_var_list.append(var)

    # optimizer for alternative update
    self.optim_e = tf.train.AdamOptimizer(learning_rate=self.lr) \
      .minimize(self.e_loss, global_step=self.step, var_list=self.encoder_var_list)
    self.optim_g = tf.train.AdamOptimizer(learning_rate=self.lr) \
      .minimize(self.g_loss, global_step=self.step, var_list=self.generator_var_list)

    # optimizer for one shot update
    self.optim = tf.train.AdamOptimizer(learning_rate=self.lr) \
      .minimize(self.loss, global_step=self.step)

    # _ = tf.scalar_summary("encoder loss", self.e_loss)
    # _ = tf.scalar_summary("generator loss", self.g_loss)
    # _ = tf.scalar_summary("total loss", self.loss)

  def build_encoder(self):
    """Inference Network. q(h|X)"""
    with tf.variable_scope("encoder1"):
      self.l1_lin = self.linear(tf.expand_dims(self.x, 0), self.embed_dim, bias=True)
      self.l1 = tf.nn.relu(self.l1_lin)
    with tf.variable_scope("encoder2"):
      self.l2_lin = self.linear(self.l1, self.embed_dim, bias=True)
      self.l2 = tf.nn.relu(self.l2_lin)
    with tf.variable_scope("encoder3"):
      self.mu = self.linear(self.l2, self.h_dim, bias=True)
    with tf.variable_scope("encoder4"):
      self.log_sigma_sq = self.linear(self.l2, self.h_dim, bias=True)

      self.eps = tf.random_normal((1, self.h_dim), 0, 1, dtype=tf.float32)
      self.sigma = tf.sqrt(tf.exp(self.log_sigma_sq))

      self.h = tf.cond(self.inference,lambda: self.mu, lambda: tf.add(self.mu, tf.multiply(self.sigma, self.eps)))


  def build_generator(self):

    """Inference Network. p(X|h)"""
    with tf.variable_scope("generator"):
      self.R = tf.get_variable("R", [self.reader.vocab_size, self.h_dim])
      self.b = tf.get_variable("b", [self.reader.vocab_size])

      self.e = -tf.matmul(self.h, self.R, transpose_b=True) + self.b
      self.p_x_i = tf.squeeze(tf.nn.softmax(self.e))

  def perplexity(self, testset, n_samples=100, print_errors=False, verbose=False):
    # Topic distribution x word distribution over topics (normalization over components is required)
    n_samples_real = n_samples
    # x_hats = np.matmul( self.lda.transform(testset[:n_samples]), (self.lda.components_ / self.lda.components_.sum(axis=1)[:, np.newaxis]))
    perplexities = []
    for i, test in enumerate(testset[:n_samples]):
      x_hat = self.p_x_i.eval(feed_dict={self.x: test['doc_tm'], self.x_idx:test['doc_tm_sparse'], self.inference: True})
      probs =  np.log(np.take(x_hat, test['doc_tm_sparse']))
      if len(probs) == 0:
        n_samples_real -= 1
        if print_errors:
          print("datapoint", i , "has no length, perplexity is now based on", n_samples_real, "samples.")
        continue

      perplexities.append( sum(probs) / len(probs))

    total_perplexity =  np.exp(-sum(perplexities) / n_samples_real)
    if verbose:
      print("nvdm perplexity on test_set",total_perplexity, "based on", n_samples_real, "samples.")
    return total_perplexity

  def experiments(self):
    self.perplexity(self.reader.test)
    self.topics()

  def calc_loss(self, dataset):
    loss = []
    prec = []
    e_loss = []
    g_loss = []
    for test in dataset:
      eloss, gloss, loss_temp, output = self.sess.run([self.e_loss, self.g_loss, self.loss, self.p_x_i],
                                                    feed_dict={self.x: test["doc_tm"],
                                                               self.x_idx:test["doc_tm_sparse"],
                                                               self.inference:True})

      loss.append( loss_temp )
      e_loss.append( eloss )
      g_loss.append( gloss )
      prec.append( self.recall(test['doc_tm'], output, n=3))

    return {'eloss': sum(e_loss) /len(e_loss),
            'rec_loss' : sum(g_loss) / len(g_loss),
            "total_loss" : sum(loss) / len(loss) ,
            "recall_3" : sum(prec) / len(prec)}

  def topics(self ,verbose=False, save=True):

    feature_names = self.reader.idx2word
    n_top_words = 10

    txt = ""
    for topic_idx in range(self.h_dim):
      z = np.zeros((1,self.h_dim))
      z[0, topic_idx] = 1
      p_x_i = self.sess.run( self.p_x_i,
                            feed_dict={self.h: z,
                                       self.inference:True})

      message = "Topic #%d: " % topic_idx
      topics = " ".join([feature_names[i]
                           for i in p_x_i.argsort()[:-n_top_words - 1:-1]])
      message += topics

      if verbose:
        print(message)

      txt += topics + '\n'

    if save:
      with open("results/nvdm_topics.txt", "w") as f:
        f.write(txt)

  def train(self):
    print("where am I?")
    tf.global_variables_initializer().run()
    # self.load(self.checkpoint_dir)

    start_time = time.time()
    start_iter = self.step.eval()

    train =  self.reader.train
    valid_set = self.reader.valid
    test_set =  self.reader.test

    #here we initialize the results file by using the labels
    with open("results/nvdm.txt", "w") as f:
      f.write("step train_perp valid_perp test_perp test_recall total_loss e_loss rec_loss\n")




    for step in range(start_iter, start_iter + self.max_iter):

      X = copy.deepcopy(train[ np.random.randint(len(train)-1, size=1)[0]])
      x = X["doc_tm"]
      x_idx = X["doc_tm_sparse"]
      """The paper update the parameters alternatively but in this repo I used oneshot update.

      _, e_loss, mu, sigma, h = self.sess.run(
          [self.optim_e, self.e_loss, self.mu, self.sigma, self.h], feed_dict={self.x: x})

      _, g_loss, summary_str = self.sess.run(
          [self.optim_g, self.g_loss, merged_sum], feed_dict={self.h: h,
                                                              self.mu: mu,
                                                              self.sigma: sigma,
                                                              self.e_loss: e_loss,
                                                              self.x_idx: x_idx})
      """

      best_perplexity = 999999999

      _, loss, mu, sigma, h, p_x_i= self.sess.run(
                                                    [self.optim, self.loss, self.mu, self.sigma, self.h, self.p_x_i],
                                                    feed_dict={self.x: x,
                                                               self.x_idx: x_idx,
                                                               self.inference:False})


      if step % 100 == 0 and False:
        print("Step: [%4d/%4d] time: %4.4f, loss: %.8f" \
              % (step, self.max_iter, time.time() - start_time, loss))

      if step % 1000 == 0:
        train_exp = self.calc_loss( train[:100])
        valid_exp = self.calc_loss( valid_set[:100])
        test_exp = self.calc_loss( test_set[:100])
        perplexity_valid =  self.perplexity(valid_set[:100])
        perplexity_train =  self.perplexity(train[:100])
        perplexity_test =  self.perplexity(test_set[:100])


        print("step:", step, "train_loss:" , train_exp["total_loss"],
              "train perplexity:", perplexity_train
              ,"valid perplexity:", perplexity_valid
              ,"test perplexity:", perplexity_test  )

        with open("results/nvdm.txt", "a") as f:
          f.write( str(step) + " "
                   +  str(perplexity_train) + " "
                   + str(perplexity_valid) + " "
                   + str(perplexity_test) + " "
                  +  str(test_exp['recall_3'] ) + " "
                  + str(train_exp["total_loss"]) + " "
                  + str(train_exp["eloss"]) + " "
                  + str(train_exp["rec_loss"]) + "\n" )







        if perplexity_valid < best_perplexity:
          best_perplexity = perplexity_valid
          # self.save(self.checkpoint_dir, step)
          self.topics( verbose=False, save=True)

# code is run using trainer.py