# ---- make vocabs from corpus base ---- #

from corpus_base import *
from helper import *

class Feature:
    def __init__(self, __data_dump):
        self.__data_dump = __data_dump
        self.__intents = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearhScreeningEvent']
        self.__vocab_set = set()

    def make_vocabs(self):
        for data in self.__data_dump:
            tokens = tokenize(data['utterance'])
            for token in tokens:
                self.__vocab_set.add(token)

        return self.__vocab_set

    
    
c = Corpus(200, 'train')
f = Feature(c.operate())

res = f.make_vocabs()
print(res)
