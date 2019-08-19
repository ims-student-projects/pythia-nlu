# ---- make vocabs from corpus base ---- #
import sys 

sys.path.append('../corpus')
from corpus_base import Corpus

sys.path.append('../utils')
from helper import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Feature:
    def __init__(self, __data_dump):
        self.__data_dump = __data_dump
        self.__intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearhScreeningEvent']
        self.__vocab_set = set()

        self.create_tfidf(self.__data_dump)

    def make_vocabs(self):
        for data in self.__data_dump:
            tokens = tokenize(data)
            for token in tokens:
                self.__vocab_set.add(token)

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
        print(X.toarray())

corpus = Corpus(3, 'train')
corpus.get_data()

all_sent = list()
for inst in corpus:
    all_sent.append(inst.get_utterance())

feature = Feature(all_sent)



# res = feature.make_vocabs()
# print(res)
