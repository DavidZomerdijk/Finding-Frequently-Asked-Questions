from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle as p
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
import copy
import random
import pickle as p
import itertools
import pandas as pd

class Reader_lstm_shortened:
    """
    This class is responsible for preprocessing the newsgroup data as well as creating batches to train.
    the input is always a list with all documents:
    """
    def __init__(self, datapath, n_features=100000, lm_minimum_freq=5, train_perc=0.6, valid_perc= 0.2, language="dutch"
                 , length_batch=5, batch_size=5, n_samples=10000, only_use_sample=True, max_sentence_length = 50):

        self.language= language
        self.lm_minimum_freq = lm_minimum_freq
        self.train_perc = train_perc
        self.valid_perc = valid_perc
        self.length_batch = length_batch
        self.batch_size = batch_size
        self.max_sentence_length = max_sentence_length

        random.seed(1)

        # self.df = self.load_data( datapath )
        # self.data  = self.shuffle( self.make_dataset( properties=properties))
        # p.dump(self.data, open("data.p", "wb"))

        # self.data = self.preprocessing_general(self.data)
        self.data = p.load(open(datapath, "rb"))
        if only_use_sample:
            self.data = self.data[:n_samples]

        self.data_prepped = self.preprocessing_tm( self.data)


        stopwords_dutch = ["aan","aangaande","aangezien","achte","achter","achterna","af","afgelopen","al","aldaar","aldus","alhoewel","alias","alle","allebei","alleen","alles","als","alsnog","altijd","altoos","ander","andere","anders","anderszins","beetje","behalve","behoudens","beide","beiden","ben","beneden","bent","bepaald","betreffende","bij","bijna","bijv","binnen","binnenin","blijkbaar","blijken","boven","bovenal","bovendien","bovengenoemd","bovenstaand","bovenvermeld","buiten","bv","daar","daardoor","daarheen","daarin","daarna","daarnet","daarom","daarop","daaruit","daarvanlangs","dan","dat","de","deden","deed","der","derde","derhalve","dertig","deze","dhr","die","dikwijls","dit","doch","doe","doen","doet","door","doorgaand","drie","duizend","dus","echter","een","eens","eer","eerdat","eerder","eerlang","eerst","eerste","eigen","eigenlijk","elk","elke","en","enig","enige","enigszins","enkel","er","erdoor","erg","ergens","etc","etcetera","even","eveneens","evenwel","gauw","ge","gedurende","geen","gehad","gekund","geleden","gelijk","gemoeten","gemogen","genoeg","geweest","gewoon","gewoonweg","haar","haarzelf","had","hadden","hare","heb","hebben","hebt","hedden","heeft","heel","hem","hemzelf","hen","het","hetzelfde","hier","hierbeneden","hierboven","hierin","hierna","hierom","hij","hijzelf","hoe","hoewel","honderd","hun","hunne","ieder","iedere","iedereen","iemand","iets","ik","ikzelf","in","inderdaad","inmiddels","intussen","inzake","is","ja","je","jezelf","jij","jijzelf","jou","jouw","jouwe","juist","jullie","kan","klaar","kon","konden","krachtens","kun","kunnen","kunt","laatst","later","liever","lijken","lijkt","maak","maakt","maakte","maakten","maar","mag","maken","me","meer","meest","meestal","men","met","mevr","mezelf","mij","mijn","mijnent","mijner","mijzelf","minder","miss","misschien","missen","mits","mocht","mochten","moest","moesten","moet","moeten","mogen","mr","mrs","mw","na","naar","nadat","nam","namelijk","nee","neem","negen","nemen","nergens","net","niemand","niet","niets","niks","noch","nochtans","nog","nogal","nooit","nu","nv","of","ofschoon","om","omdat","omhoog","omlaag","omstreeks","omtrent","omver","ondanks","onder","ondertussen","ongeveer","ons","onszelf","onze","onzeker","ooit","ook","op","opnieuw","opzij","over","overal","overeind","overige","overigens","paar","pas","per","precies","recent","redelijk","reeds","rond","rondom","samen","sedert","sinds","sindsdien","slechts","sommige","spoedig","steeds","tamelijk","te","tegen","tegenover","tenzij","terwijl","thans","tien","tiende","tijdens","tja","toch","toe","toen","toenmaals","toenmalig","tot","totdat","tussen","twee","tweede","u","uit","uitgezonderd","uw","vaak","vaakwat","van","vanaf","vandaan","vanuit","vanwege","veel","veeleer","veertig","verder","verscheidene","verschillende","vervolgens","via","vier","vierde","vijf","vijfde","vijftig","vol","volgend","volgens","voor","vooraf","vooral","vooralsnog","voorbij","voordat","voordezen","voordien","voorheen","voorop","voorts","vooruit","vrij","vroeg","waar","waarom","waarschijnlijk","wanneer","want","waren","was","wat","we","wederom","weer","weg","wegens","weinig","wel","weldra","welk","welke","werd","werden","werder","wezen","whatever","wie","wiens","wier","wij","wijzelf","wil","wilden","willen","word","worden","wordt","zal","ze","zei","zeker","zelf","zelfde","zelfs","zes","zeven","zich","zichzelf","zij","zijn","zijne","zijzelf","zo","zoals","zodat","zodra","zonder","zou","zouden","zowat","zulk","zulke","zullen","zult"]

        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words=stopwords_dutch)

        #first fit the matrix on the train set
        self.data_tm = [ x["data_gp"] for x in self.data]
        # print( self.data[:10] )
        self.tf_vectorizer.fit_transform( self.data_tm[:int(len(self.data_tm)*train_perc)])
        self.tf_data = self.reluDerivative( self.tf_vectorizer.transform(self.data_tm))

        self.idx2word = self.tf_vectorizer.get_feature_names()
        self.vocab_size = np.shape(self.tf_data)[1]
        print("vocab_size", self.vocab_size)


        #LM data
        self.train, self.valid, self.test, self.lm_id2word, self.lm_word2id, self.lm_vocab_size = self.preprocessing_lm (data=self.data)


    def load_data(self, filepath):
        df_raw = pd.read_csv(filepath, sep=';', encoding = "ISO-8859-1")

        columns_keep = ['id', 'omschrijving']

        df = df_raw[columns_keep]

        #here we confert the registration date to useful numbers
        df["reg_day"] = df["datum"].astype(str).str[:2].astype(int)
        months_dict = {"JAN" : 1, "FEB": 2, "MAR" : 3, "APR" : 4, "MAY": 5, "JUN" : 6 , "JUL":7, "AUG": 8, "SEP": 9 , "OKT":10, "NOV": 11, "DEC":12}
        df["reg_month"] =  df["datum"].apply(lambda x: months_dict[ x[2:5] ]).astype(int)
        df["reg_year"] =  df["datum"].astype(str).str[5:9].astype(int)
        dates = pd.DataFrame( {"year": df['reg_year'],
                               "month": df['reg_month'],
                               "day": df[ 'reg_day']})
        df["reg_date"] = pd.to_datetime( dates )
        #here we only select one sr_id
        df.sort_values(by='reg_date',ascending=True).reset_index(drop=True)
        df = df.groupby("sr_id").first().reset_index()
        return df

    def make_dataset(self, properties):
        data = self.df[properties].T.to_dict().values()
        new_data = []
        for x in data:
            sentence = x['omschrijving']
            try:
                new_data.append({"raw": x["omschrijving"] ,"omschrijving": word_tokenize(sentence) })
            except:
                print("exception occured, the following document was found in the dataset:",sentence)

        return new_data

    def shuffle(self,x):
        x_new = [[doc] for doc in x]
        random.shuffle( x_new)
        return [x[0] for x in x_new]

    # takes data in the form of list of strings
    def preprocessing_lm(self, data, source_field='data_gp'):
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


        data_listform = [[word_tokenize(y, language=self.language) for y in sent_tokenize(x[source_field], language=self.language)]for x in data ]
        #get tf for train set

        with open('corpus.0', 'w') as f:
            for doc in data_listform:
                new_doc = [item for sublist in doc for item in sublist]
                if len(new_doc) <= self.max_sentence_length:
                    doc = " ".join([item for sublist in doc for item in sublist])
                    f.write(doc + "\n")

        tf_train = get_tf(data_listform[:int(len(data_listform)*self.train_perc)])
        data_listform = remove_numbers(data_listform)
        data_listform = remove_rare_words(data_listform, tf_train, min_freq=self.lm_minimum_freq)

        idx2word, word2idx = create_vocab(data_listform)

        tokenized_data = [[[word2idx[word] for word in sen]for sen in doc ] for doc in data_listform]

        language_model_data = create_language_model_data(tokenized_data, word2idx)
        # the data which we will use for our extention

        new_tf = copy.deepcopy(self.tf_data)
        new_data_set = [ {"doc_tm" :x, "doc_tm_sparse" : np.where(x>0)[0], "doc_lm": get_batch_data( language_model_data[i] ),
                          "lstm_model": language_model_data[i]}  for i, x in enumerate(new_tf)
                         if len(np.where(x>0)[0]) > 0 and language_model_data[i]!=None and get_batch_data( language_model_data[i] ) != None and len(language_model_data[i]) <=self.max_sentence_length]

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
    def preprocessing_general(self, data, remove_the_uppercase = True, remove_the_numbers=True, stem=False, lemmatize=False):

        def remove_uppercase(data):
            for x in data:
                x["data_gp"] = x["raw"].lower()
            return data


        def remove_numbers(d):
            for docs in d:
                datapoint = docs['data_gp']
                tokenized = [word_tokenize(y, language=self.language) for y in sent_tokenize(datapoint, language=self.language)]
                removed =  [[word if not word.isdigit() else "<NUMBER>" for word in sen]for sen in tokenized]
                docs['data_gp'] = " ".join([ " ".join([word for word in s]) for s in removed ])
            return d

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


    def reluDerivative(self, input):
        x = input.toarray()
        x[x<=0] = 0
        x[x>0] = 1
        return x


if __name__ == "__main__":

    reader = Reader_lstm_shortened( datapath="data.p", n_features=1000, lm_minimum_freq=5, train_perc=0.6, valid_perc= 0.2,
                                   language="dutch",
                                   length_batch=5,
                                   batch_size=5,
                                   n_samples=50000,
                                   only_use_sample=True,max_sentence_length = 30)


    print("success!")








