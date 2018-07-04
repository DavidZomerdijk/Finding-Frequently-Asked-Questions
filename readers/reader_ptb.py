
import collections
import os
import sys
import copy
import tensorflow as tf
import numpy as np



class Reader_PTB:
    """
    This class is responsible for preprocessing the PTB data as well as creating batches to train.
    the input is always a list with all documents:
    """
    def __init__(self, datapath, length_batch, batch_size):
        self.length_batch = length_batch
        self.batch_size = batch_size
        raw_data = self.ptb_raw_data(data_path=datapath)
        self.train_data, self.valid_data, self.test_data, self.word2id = raw_data
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.lm_vocab_size = len(self.word2id)
        self.train_data = self.get_batch_data(self.train_data)
        self.valid_data = self.get_batch_data(self.valid_data)
        self.test_data = self.get_batch_data(self.test_data)
        print(len(self.train_data))

    def _read_words(self, filename):
        with tf.gfile.GFile(filename, "r") as f:
                return f.read().replace("\n", "<eos>").split()

    def _build_vocab(self, filename):
        data = self._read_words(filename)

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        return word_to_id

    def _file_to_word_ids(self, filename, word_to_id):
        data = self._read_words(filename)
        return [word_to_id[word] for word in data if word in word_to_id]




    def ptb_raw_data(self, data_path=None):
        """Load PTB raw data from data directory "data_path".

        Reads PTB text files, converts strings to integer ids,
        and performs mini-batching of the inputs.

        The PTB dataset comes from Tomas Mikolov's webpage:

        http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

        Args:
          data_path: string path to the directory where simple-examples.tgz has
            been extracted.

        Returns:
          tuple (train_data, valid_data, test_data, vocabulary)
          where each of the data objects can be passed to PTBIterator.
        """

        train_path = os.path.join(data_path, "ptb.train.txt")
        valid_path = os.path.join(data_path, "ptb.valid.txt")
        test_path = os.path.join(data_path, "ptb.test.txt")

        word_to_id = self._build_vocab(train_path)
        train_data = self._file_to_word_ids(train_path, word_to_id)
        valid_data = self._file_to_word_ids(valid_path, word_to_id)
        test_data = self._file_to_word_ids(test_path, word_to_id)

        return train_data, valid_data, test_data, word_to_id

    def get_batch_data(self, data):
        def create_batches(d,batch_size=1, lstm_length=20):
            batches = len(d) // (lstm_length * batch_size)
            cutoff = batches * lstm_length * batch_size
            d = np.array(d[:cutoff])
            # for larger batch size
            d = d.reshape((batch_size, batches * lstm_length))
            # horizontal split
            output = np.hsplit(d, [i*lstm_length for i in range(1, batches)])
            # output = d.reshape(-1, 1, lstm_length)
            return output


        x = copy.deepcopy(data[:-1])
        y = copy.deepcopy(data[1:])
        x_batch = create_batches(x, batch_size=self.batch_size, lstm_length=self.length_batch)
        y_batch = create_batches(y, batch_size=self.batch_size, lstm_length=self.length_batch)
        return [(x_batch[i], y_batch[i])  for i in range(len(x_batch))]


if __name__ == "__main__":
    reader = PTB(datapath="../data/", length_batch=35, batch_size=20)
    print( len(reader.train_data))