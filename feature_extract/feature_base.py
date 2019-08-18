# ---- make vocabs from corpus base ---- #
import sys 

sys.path.append('../corpus')
from corpus_base import Corpus

sys.path.append('../utils')
import helper


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

    
    
corpus = Corpus(200, 'train')
corpus.get_data()





# res = f.make_vocabs()
# print(res)
