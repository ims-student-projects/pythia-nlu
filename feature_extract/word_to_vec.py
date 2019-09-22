import numpy as np

import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet as wn
tknzr = TweetTokenizer()

# from helper import *

import pandas as pd
import os
import numpy as np
from numpy import zeros

# create vocabulary
  
##### create the word2vec dict from the dictionary #####

# embedding_path = "/home/users0/sengupmt/Dokumente/glove.6B.300d.txt" 

def get_word2vec(embedding_path):
    file = open(embedding_path, "r")
    if (file):
        word2vec = dict()
        split = file.read().splitlines()
        for line in split:
            key = line.split(' ',1)[0] # the first word is the key
            value = np.array([float(val) for val in line.split(' ')[1:]])
            word2vec[key] = value
        return (word2vec)
    else:
        print("Invalid file path")

