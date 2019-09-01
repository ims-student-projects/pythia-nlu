# ---- make vocabs from corpus base ---- #
import sys 

sys.path.append('../corpus')
from corpus_base import Corpus

sys.path.append('../utils')
from helper import *

sys.path.append('../models')
from baseline_svm_intent import SVM

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Feature:
    def __init__(self):
        # self.__data_dump = __data_dump
        self.__intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearhScreeningEvent']
        self.__vocab_set = set()

        # self.create_tfidf(self.__data_dump)

    # def make_vocabs(self):
    #     for data in self.__data_dump:
    #         tokens = tokenize(data)
    #         for token in tokens:
    #             self.__vocab_set.add(token)

        # return self.__vocab_set

    def create_tfidf(self, __all_sents):
        # self.__all_vocabs = __all_vocabs
        self.__all_sents = __all_sents

        # # ---- Taking into account just the term frequencies:
        # vectorizer = CountVectorizer(ngram_range=(2,2))
        # # ---- The ngram range specifies the ngram configuration.

        # X = vectorizer.fit_transform(self.__all_sents)
        
        # # ---- Testing the ngram generation:
        # # ---------------- #
        # print(vectorizer.get_feature_names())
        # # ---------------- #
        # print(X.toarray())

        vectorizer = TfidfVectorizer(ngram_range=(2,2)) # You can still specify n-grams here.
        X = vectorizer.fit_transform(self.__all_sents)

        # # ---- Testing the TFIDF value + ngrams:
        # print(X.toarray())

        return X.toarray()

corpus_tr = Corpus(1950, 'train')
corpus_tr.get_data()

corpus_ts = Corpus(30, 'test')
corpus_ts.get_data()

# ---- grab all utterances ----
all_sent_tr = list()
for inst in corpus_tr:
    all_sent_tr.append(inst.get_utterance())

# ---- grab all utterances ----
all_sent_ts = list()
for inst in corpus_ts:
    all_sent_ts.append(inst.get_utterance())

# ---- train feature ----
feature_tr = Feature()

feature_tr = feature_tr.create_tfidf(all_sent_tr)

# ---- test feature ---- 
feature_ts = Feature()
feature_ts = feature_ts.create_tfidf(all_sent_ts)

# print(feature_ts.create_tfidf(all_sent_ts))

# ---- grab all train targets ----
all_targ_tr = list()
for inst in corpus_tr:
    all_targ_tr.append(inst.get_gold_intent())

# print(all_targ_tr)

# ---- grab all test targets ----
all_targ_ts = list()
for inst in corpus_ts:
    all_targ_ts.append(inst.get_gold_intent())


# --------------------------------------- #

# all_data = feature_tr.append(feature_ts)
# all_targ = all_targ_tr + all_targ_ts

# ------------- enter all data and all target as input to SVM ----------------- #

baseline_svm = SVM(feature_tr, all_targ_tr)

baseline_svm.train()
