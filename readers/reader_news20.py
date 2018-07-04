from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle as p
import numpy as np
from random import randint
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
import copy
import random
import itertools

class Reader_News20:
    """
    This class is responsible for preprocessing the newsgroup data as well as creating batches to train.
    the input is always a list with all documents:
    """
    def __init__(self, data, n_features=1000, lm_minimum_freq=5, train_perc=0.6, valid_perc= 0.2, language="english"
                 , length_batch=5, batch_size=5 ):
        #data preprocessing
        #todo: remove limiting number of samples
        random.seed(1)

        self.language= language
        self.lm_minimum_freq = lm_minimum_freq
        self.train_perc = train_perc
        self.valid_perc = valid_perc
        self.length_batch = length_batch
        self.batch_size = batch_size

        self.data_samples = self.preprocessing_general( self.shuffle( data ))
        self.data_tm = self.preprocessing_tm( self.data_samples)


        #use for ntm model
        self.data_prepped = [self.process_doc(doc, i) for i, doc in enumerate(self.data_samples)]

        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words=self.language)

        # self.tf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
        #                                      max_features=n_features,
        #                                      stop_words=self.language)

        #first fit the matrix on the train set
        self.tf_vectorizer.fit_transform( self.data_tm[:int(len(self.data_tm)*train_perc)])
        self.tf = self.reluDerivative( self.tf_vectorizer.transform(self.data_tm))

        self.idx2word = self.tf_vectorizer.get_feature_names()
        self.vocab_size = np.shape(self.tf)[1]

        #LM data
        self.train, self.valid, self.test, self.lm_id2word, self.lm_word2id, self.lm_vocab_size = self.preprocessing_lm(data=self.data_samples, minimum_tf=lm_minimum_freq)

    def shuffle(self,x):
        x_new = [[doc] for doc in x]
        random.shuffle( x_new)
        return [x[0] for x in x_new]

    # takes data in the form of list of strings
    def preprocessing_lm(self, data, minimum_tf):
        # gets tf from corpus
        def get_tf(d):
            tf = defaultdict(int)
            for doc in d:
                for sen in doc:
                    for word in sen:
                        tf[word]+=1
            return tf

        def create_vocab(data):
            idx2word = []
            word2idx = dict()
            for doc in data:
                for sen in doc:
                    for word in sen:
                        if word not in word2idx:
                            word2idx[word] = len(idx2word)
                            idx2word.append(word)
            word2idx["<EOS>"] = len( idx2word )
            idx2word.append("<EOS>")
            word2idx["<BOS>"] = len( idx2word)
            idx2word.append("<BOS>")
            word2idx["<PAD>"] = len( idx2word)
            idx2word.append("<PAD>")
            return idx2word, word2idx


        def remove_numbers(data):
            return [[[word if not word.isdigit() else "<NUMBER>" for word in sen]for sen in doc] for doc in data]

        # removes rare words
        def remove_rare_words(data, tf, min_freq):
            return [[[word if tf[word] >= min_freq else "<UNK>" for word in sen]for sen in doc] for doc in data]


        def create_language_model_data(data, word2idx):
            lm_data = []
            for doc in data:
                if doc == []:
                    lm_data.append(None)
                    continue
                doc_new = [copy.deepcopy(sen) for sen in doc]
                doc_new[0].insert(0, word2idx["<EOS>"] )

                for sen in doc_new:
                    sen.append( word2idx["<EOS>"] )
                lm_data.append(doc_new)

                # print( lm_data)
            lm_data = [ list(itertools.chain.from_iterable(doc)) if doc != None else None for doc in lm_data]
            return lm_data

        def get_batch_data( data):
            def create_batches(d,batch_size=1, lstm_length=20):
                batches = len(d) // (lstm_length * batch_size)
                if batches == 0:
                    # print( "peep peep")
                    return None
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
            if x_batch == None:
                return None

            return [(x_batch[i], y_batch[i])  for i in range(len(x_batch))]


        data_listform = [[word_tokenize(y, language=self.language) for y in sent_tokenize(x, language=self.language)]for x in data ]
        #get tf for train set
        # with open('coherence_data/news20/corpus.0', 'w') as f:
        #     for doc in data_listform:
        #         doc = " ".join([item for sublist in doc for item in sublist])
        #         f.write(doc + "\n")

        tf_train = get_tf(data_listform[:int(len(data_listform)*self.train_perc)])
        data_listform = remove_numbers(data_listform)
        data_listform = remove_rare_words(data_listform, tf_train, min_freq=self.lm_minimum_freq)
        idx2word, word2idx = create_vocab(data_listform)

        tokenized_data = [[[word2idx[word] for word in sen]for sen in doc ] for doc in data_listform]

        language_model_data = create_language_model_data(tokenized_data, word2idx)

        new_tf = copy.deepcopy(self.tf)
        new_data_set = [ {"doc_tm" :x, "doc_tm_sparse" : np.where(x>0)[0], "doc_lm": get_batch_data( language_model_data[i] )}  for i, x in enumerate(new_tf)
                         if len(np.where(x>0)[0]) > 0 and language_model_data[i]!=None and get_batch_data( language_model_data[i] ) != None ]
        total_length = len(new_data_set)
        train_idx = int(total_length*self.train_perc)
        valid_idx = int(total_length*(self.train_perc+self.valid_perc))
        train =  new_data_set[:train_idx]
        valid = new_data_set[train_idx:valid_idx]
        test =  new_data_set[valid_idx:]

        return train, valid, test, idx2word, word2idx, len(idx2word)


    def get_sets(self, valid_perc=0.2):
        new_tf = copy.deepcopy(self.tf)
        # here we add the indices that are on and remove documents that contain no words that are in te vocab
        # the third variable is the text
        new_data_set = [ {"doc_tm": x, "doc_tm_1" : np.where(x>0)[0], "doc_lm" : self.language_model_data[i] }  for i, x in enumerate(new_tf)
                         if len(np.where(x>0)[0]) > 0 and self.language_model_data[i]!=None ]
        total_length = len(new_data_set)
        train_idx = int(total_length*self.train_perc)
        valid_idx = int(total_length*(self.train_perc+valid_perc))

        train =  new_data_set[:train_idx]
        valid = new_data_set[train_idx:valid_idx]
        test =  new_data_set[valid_idx:]
        return train, valid, test

    # removes lowercase, lemmatize? , stem?
    def preprocessing_general(self, data, remove_the_uppercase = True, remove_the_numbers=False, stem=False, lemmatize=False):

        def remove_uppercase(data):
            new_data = []
            for x in data:
                new_data.append( x.lower() )
            return new_data


        def remove_numbers(d):
            new_data = [[word_tokenize(y, language=self.language) for y in sent_tokenize(x, language=self.language)]for x in d]
            data_no_digits =  [[[word if not word.isdigit() else "<NUMBER>" for word in sen]for sen in doc] for doc in new_data]

            return [ " ".join([ " ".join([word for word in s]) for s in doc ])for  doc in data_no_digits]

        new_data = data
        if remove_the_uppercase:
            print("replacing uppercase by lowercase")
            new_data = remove_uppercase(new_data)

        if remove_the_numbers:
            print("removing numbers from general data")
            new_data = remove_numbers(new_data)

        return new_data


    def preprocessing_tm(self, data):
        return data

    def process_doc(self, doc, i):
        """"this function preprocesses the documents
        """

        sentences = sent_tokenize(doc)
        output_data = [word_tokenize(s) for s in sentences ]
        return output_data

    def reluDerivative(self, input):
        x = input.toarray()
        x[x<=0] = 0
        x[x>0] = 1
        return x



if __name__ == "__main__":
    dataset = p.load(open("../data/news20/news20.p" , "rb"))
    reader = Reader_News20(dataset)

    # p.dump(reader, open("../saved/news20reader.p", "wb"))
    # reader = p.load(open("../saved/news20reader.p", "rb"))
    print( reader.train[0])





