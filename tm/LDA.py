import time
import numpy as np
import tensorflow as tf
from base import Model
from collections import defaultdict
import copy
import itertools
import math
import scipy.special
import sys
from sklearn.decomposition import  LatentDirichletAllocation
sys.path.append("../readers")
import pickle
from reader_news20 import Reader_News20
from reader_apnews import Reader_APNEWS
# from reader import TextReader

class LDA(Model):
    def __init__(self, reader, dataset='', topics=50, max_iter=20):

        self.lda = LatentDirichletAllocation(n_components=topics, max_iter=max_iter,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        self.reader = reader
        self.dataset = dataset
        self.n_topics = topics

        self.train = np.array([x["doc_tm"] for x in self.reader.train])
        self.valid =  np.array([x["doc_tm"] for x in self.reader.valid])
        self.test = np.array([x["doc_tm"] for x in self.reader.test])


        self.lda.fit(self.train)

    def show_topics(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

    def save_topic_distribution(self, save_path, n_top_words):
        model = self.lda
        feature_names = self.reader.idx2word
        str = ""

        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            str += message + "\n"
        str += "\n"
        print(str)
        with open( save_path, "a" ) as f:
            f.write(str)

        print("saved to", save_path)

    def LDA_recall(self, testset, n_samples=1000, n_recall=3, print_output=True):
        x = testset[:n_samples]
        t =  self.lda.transform( x )
        x_hat = np.matmul( t, self.lda.components_ )

        recall_tot = []
        for i in range(np.shape(x)[0]):
            x_temp = x[i ,:]
            x_hat_temp = x_hat[i,:]
            recall_tot.append(self.recall(x_temp, x_hat_temp, n_recall))

        output = np.sum(recall_tot)/ len(recall_tot)
        if print_output:
            print("recall", n_recall, "over", n_samples, ":" , output)
        return output

    def get_topic_distribution(self,x):
        return self.lda.transform(x)

    def perplexity(self, testset, n_samples=100, print_errors=False):
        # Topic distribution x word distribution over topics (normalization over components is required)
        n_samples_real = n_samples
        x_hats = np.matmul( self.lda.transform(testset[:n_samples]), (self.lda.components_ / self.lda.components_.sum(axis=1)[:, np.newaxis]))
        perplexities = []

        for i in range(n_samples):
            idxs = np.where(testset[i] > 0)

            x_hat = x_hats[i]
            probs =  np.log(np.take(x_hat, idxs))
            if len(probs[0]) == 0:
                n_samples_real -= 1
                if print_errors:
                    print("datapoint", i , "has no length, perplexity is now based on", n_samples_real, "samples.")
                continue

            perplexities.append( sum(probs[0]) / len(probs[0]))

        total_perplexity =  np.exp(-sum(perplexities) / n_samples_real)#np.exp(- sum(perplexities) / len(perplexities))
        print("LDA perplexity on test_set",total_perplexity)



    def experiments(self,  save_location="topics.txt"):
        self.show_topics(self.lda, self.reader.idx2word, 10)
        feature_names = self.reader.idx2word
        n_top_words = 10
        self.LDA_recall( self.test ,print_output=True)
        self.perplexity(self.test)
        self.save_topic_distribution( save_location, 10)



if __name__ == '__main__':

    # news20
    # dataset = pickle.load(open("../data/news20/news20.p" , "rb"))
    # reader = Reader_News20(dataset, n_features=1000)
    # pickle.dump(reader, open("../saved/news20reader.p", "wb"))
    # reader = pickle.load(open("../saved/news20reader.p", "rb"))

    # APnews
    # reader = Reader_APNEWS(datapath="../data/apnews/apnews.dat", n_features=1000, sample_size=10000)
    # pickle.dump(reader, open("../saved/apnews.p", "wb"))
    reader = pickle.load(open("../saved/apnews.p", "rb"))

    model = LDA(reader  , topics=50  ,max_iter=30  )
    model.experiments()


    #
    # # Penn treebank
    # data_path = "../data/ptb"
    # reader = TextReader(data_path)
    # model = LDA(reader  )
    # model.experiments()
