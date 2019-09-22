
# ---- make vocabs from corpus base ---- #
import sys 
sys.path.append(sys.path[0] + '/../')
# from corpus.corpus_base import Corpus
# from utils.helper import *
# from models.baseline_svm_intent import SVM
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.tokenize import word_tokenize

class Feature:
    def __init__(self, vocab):
        # self.__data_dump = __data_dump
        self.__intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearhScreeningEvent']
        self.__vocab_set = vocab
        self.__vectorizer = TfidfVectorizer(ngram_range=(1,1), vocabulary=self.__vocab_set) # You can still specify n-grams here.

        # self.create_tfidf(self.__data_dump)

    # def make_vocabs(self):
    #     for data in self.__data_dump:
    #         tokens = tokenize(data)
    #         for token in tokens:
    #             self.__vocab_set.add(token)

        # return self.__vocab_set

    def get_tokens(all_sent):
        all_tokens=list()
        for sent in all_sent:
            sent = word_tokenize(sentence)
            for word in sentence:
                all_tokens.append(word)
        return all_tokens


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

        X = self.__vectorizer.fit_transform(self.__all_sents)
        print('tfidfvectorizer vocab: ', len(self.__vectorizer.vocabulary))
        print('tfidfvectorizer vocab_: ', len(self.__vectorizer.vocabulary_))
        print('TYPE OF X: ', type(X))
        print('SHAPE OF X: ', X.shape)
        # # ---- Testing the TFIDF value + ngrams:
        # print(X.toarray())

        return X.toarray()


