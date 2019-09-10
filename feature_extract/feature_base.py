
# ---- make vocabs from corpus base ---- #
import sys 
sys.path.append(sys.path[0] + '/../')
from corpus.corpus_base import Corpus
from utils.helper import *
from models.baseline_svm_intent import SVM
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

corpus_tr = Corpus(1500, 'train')
corpus_tr.shuffle()

corpus_ts = Corpus(200, 'test')
corpus_ts.shuffle()


# ---- grab all utterances ----
all_sent_tr = list()
for inst in corpus_tr:
    all_sent_tr.append(inst.get_utterance())

# ---- grab all utterances ----
all_sent_ts = list()
for inst in corpus_ts:
    all_sent_ts.append(inst.get_utterance())

all_sent_combined = all_sent_tr + all_sent_ts
vocab = set()
for sent in all_sent_combined:
    vocab.update(sent.split())

# ---- train feature ----
feat = Feature(vocab)

feature_tr = feat.create_tfidf(all_sent_tr)

# ---- test feature ---- 
feature_ts = feat.create_tfidf(all_sent_ts)

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

baseline_svm = SVM(feature_tr, all_targ_tr, feature_ts, all_targ_ts)

baseline_svm.train()
