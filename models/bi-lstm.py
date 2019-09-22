import sys 
sys.path.append(sys.path[0] + '/../')
from corpus.corpus_base import Corpus
from feature_extract.feature_base import Feature

import numpy as np

from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.layers import TimeDistributed

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

        self.setup()
        self.train_data = self.all_sent_tr
        self.train_targets = self.all_targ_tr
        self.test_data = self.all_sent_ts
        self.test_targets = self.all_targ_ts

        self.embedding_path = "/home/users0/sengupmt/Dokumente/Moody/glove.6B.50d.txt"

        # self.intent_ls = emotion_ls
        # self.token_ls = token_ls
        # self.utterances = utterances
        # self.max_len = max_len

    def train(self):
        ##### OUTPUT VARIABLE #####
        
        le = preprocessing.LabelEncoder()
        self.Y_new_tr = self.all_targ_tr  
        self.Y_new_tr = le.fit_transform(self.Y_new_tr)
        if self.Y_new_tr.size > 0:
            self.inverted_label = le.inverse_transform(self.Y_new_tr)

        self.Y_new_ts = self.all_targ_ts  
        self.Y_new_ts = le.fit_transform(self.Y_new_ts)
        if self.Y_new_ts.size > 0:
            self.inverted_label = le.inverse_transform(self.Y_new_ts)
        
        ###### INPUT VARIABLE ######

        # prepare tokenizer
        self.t = Tokenizer()
        self.t.fit_on_texts(self.token_ls)
        self.vocab_size = len(self.t.word_index) + 1

        encoded_docs_tr = self.t.texts_to_sequences(self.all_sent_tr)
        encoded_docs_ts = self.t.texts_to_sequences(self.all_sent_ts) 

        max_length = self.max_len
        self.X_train = pad_sequences(encoded_docs_tr, maxlen=max_length, padding='post')
        self.X_test = pad_sequences(encoded_docs_ts, maxlen=max_length, padding='post')
        self.Y_train = self.Y_new_tr
        self.Y_train = self.Y_new_ts

        # get the embedding matrix from the embedding layer

        self.embedding_matrix = zeros((self.vocab_size, 50))
        w2v = get_word2vec(self.embedding_path)
        for word, i in self.t.word_index.items():
            embedding_vector = w2v.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

        # Splitting into test and training data

        # self.X_train,self.X_test, self.Y_train, self.Y_test =  train_test_split(self.X, self.y,test_size =0.20,random_state= 4)

        # can pass all these variables as parameters to the model function

        # main model
    def rnn_model(self):
        self.model = Sequential()
        input_initial = Input(shape=(self.max_len,))
        self.model = Embedding(self.vocab_size,50,weights=[self.embedding_matrix],input_length=self.max_len)(input_initial)
        self.model =  Bidirectional (LSTM (50,return_sequences=True,dropout=0.20),merge_mode='concat')(self.model)
        self.model = TimeDistributed(Dense(50,activation='relu'))(self.model)
        self.model = Flatten()(self.model)
        self.model = Dense(50,activation='relu')(self.model)
        output = Dense(7,activation='softmax')(self.model)
        self.model = Model(input_initial,output)
        self.model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


        self.model.fit(self.X_train,self.Y_train)

        return self.evaluate()

    def evaluate(self):
        # evaluate the model
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=2)

        self.Y_pred = self.model.predict(self.X_test)
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

        self.token_ls = feat.get_tokens(all_sent_combined)

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
    tr = Corpus(10,'train')
    ts = Corpus(2, 'test')
    bilstm = BiLSTM(tr,ts)
    bilstm.train()
    bilstm.rnn_model()