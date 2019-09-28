import sys 
sys.path.append(sys.path[0] + '/../')
from corpus.corpus_base import Corpus
from feature_extract.feature_base import Feature

import numpy as np

from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet as wn
tknzr = TweetTokenizer()

# from helper import *

import pandas as pd
import os
import numpy as np
from numpy import zeros

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing

from feature_extract.word_to_vec import *



############################################## Bi LSTM with word embeddings ##########################################################

class BiLSTM:
    def __init__(self, train_corpus, test_corpus, max_len=40):

        self.corpus_tr = train_corpus
        self.corpus_ts = test_corpus
        self.__predicted = []

        self.max_length = max_len
        self.setup()
        self.train_data = self.all_sent_tr
        self.train_targets = self.all_targ_tr
        self.test_data = self.all_sent_ts
        self.test_targets = self.all_targ_ts

        self.embedding_path = "glove.6B.300d.txt"

    def train(self):
        ##### OUTPUT VARIABLE #####
        
        self.le = preprocessing.LabelEncoder()
        self.Y_new_tr = self.all_targ_tr  
        self.Y_new_tr = self.le.fit_transform(self.Y_new_tr)
        
        self.Y_new_ts = self.all_targ_ts  
        self.Y_new_ts = self.le.fit_transform(self.Y_new_ts)
        
        ###### INPUT VARIABLE ######

        # prepare tokenizer
        self.t = Tokenizer()
        self.t.fit_on_texts(self.token_ls)
        self.vocab_size = len(self.t.word_index) + 1

        encoded_docs_tr = self.t.texts_to_sequences(self.all_sent_tr)
        encoded_docs_ts = self.t.texts_to_sequences(self.all_sent_ts) 

        self.X_train = pad_sequences(encoded_docs_tr, maxlen=self.max_length, padding='post')
        self.X_test = pad_sequences(encoded_docs_ts, maxlen=self.max_length, padding='post')
        self.Y_train = self.Y_new_tr
        self.Y_test = self.Y_new_ts

        # get the embedding matrix from the embedding layer

        self.embedding_matrix = zeros((self.vocab_size, 300))
        w2v = get_word2vec(self.embedding_path)
        for word, i in self.t.word_index.items():
            embedding_vector = w2v.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

        # main model
    def rnn_model(self):
        self.model = Sequential()
        input_initial = Input(shape=(self.max_length,))
        self.model = Embedding(self.vocab_size,300,weights=[self.embedding_matrix],input_length=self.max_length)(input_initial)
        self.model =  Bidirectional (LSTM (300,return_sequences=True,dropout=0.20),merge_mode='concat')(self.model)
        self.model = TimeDistributed(Dense(300,activation='relu'))(self.model)
        self.model = Flatten()(self.model)
        self.model = Dense(300,activation='relu')(self.model)
        output = Dense(7,activation='softmax')(self.model)
        self.model = Model(input_initial,output)
        self.model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


        self.model.fit(self.X_train,self.Y_train)
        # self.model.save('lstm_model.h5')
        return self.evaluate()

    def evaluate(self):
        # evaluate the model
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=2)

        self.Y_pred = self.model.predict(self.X_test)
        
        __prbList = list()

        for p in self.Y_pred:
            probs = {}
            for i in range(len(p)):
                probs[self.le.inverse_transform([i])[0]] = p[i]
            __prbList.append(probs)
        print(__prbList)
        print('CORPUS_TS size: ', self.corpus_ts.get_size())        
        
        for i, j in zip(__prbList, self.corpus_ts):
            j.set_intent_probabilities(i)

        print('----------- Intent probabilities set is complete ---------------')

        self.y_pred = np.array([np.argmax(pred) for pred in self.Y_pred])

        
        
        print('Result:\n',classification_report(self.Y_test,self.y_pred),'\n')

    def setup(self):
        # ---- grab all utterances ----
        self.all_sent_tr = list()
        for inst in self.corpus_tr:
            self.all_sent_tr.append(inst.get_utterance())

        # ---- grab all utterances ----
        self.all_sent_ts = list()
        for inst in self.corpus_ts:
            self.all_sent_ts.append(inst.get_utterance())

        self.all_sent_combined = self.all_sent_tr + self.all_sent_ts
        self.vocab = set()
        for sent in self.all_sent_combined:
            self.vocab.update(sent.split())

        # ---- train feature ----
        feat = Feature(self.vocab)

        self.token_ls = feat.get_tokens(self.all_sent_combined)

        self.feature_tr = feat.create_tfidf(self.all_sent_tr)

        # ---- test feature ---- 
        self.feature_ts = feat.create_tfidf(self.all_sent_ts)

        # print(feature_ts.create_tfidf(all_sent_ts))

        # ---- grab all train targets ----
        self.all_targ_tr = list()
        for inst in self.corpus_tr:
            self.all_targ_tr.append(inst.get_gold_intent())

        # print(all_targ_tr)

        # ---- grab all test targets ----
        self.all_targ_ts = list()
        for inst in self.corpus_ts:
            self.all_targ_ts.append(inst.get_gold_intent())

if __name__ == '__main__':
    tr = Corpus(1400,'train')
    ts = Corpus(20, 'test')
    bilstm = BiLSTM(tr,ts)
    bilstm.train()
    bilstm.rnn_model()