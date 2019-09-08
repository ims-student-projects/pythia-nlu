import sys
import random

sys.path.append(sys.path[0] + '/../')

from utils.helper import *
from corpus.query import *

class Corpus:
    def __init__(self, size, type_req):
        # private attributes
        self.__corpus = list()
        self.__curr = 0 # counter for iterator
        # public attributes
        self.size = size
        self.type_req = type_req
        self.intent_labels = {}
        self.slot_labels = []
        self.file_paths = [ 'slp_train_add_to_playlist_full.txt', 
                            'slp_train_book_restaurant_full.txt', 
                            'slp_train_get_weather_full.txt', 
                            'slp_train_play_music_full.txt', 
                            'slp_train_rate_book_full.txt', 
                            'slp_train_search_creative_work_full.txt', 
                            'slp_train_search_screening_event_full.txt' ]
        # import data from files
        self.get_data()


    def __next__(self):
        if self.__curr >= self.length():
            raise StopIteration
        else:
            self.__curr += 1
            return self.__corpus[self.__curr - 1]


    def __iter__(self):
        return iter(self.__corpus)


    def shuffle(self):
        random.shuffle(self.__corpus)


    def get_labels(self):
        if not self.intent_labels or not self.slot_labels:
            for x in self.__corpus:
                intent = x.get_gold_intent()
                slots = x.get_gold_slots()
                if intent not in self.intent_labels:
                    self.intent_labels[intent] = []
                for slot in slots:
                    if slot not in self.intent_labels[intent]:
                        self.intent_labels[intent].append(slot)
                    if slot not in self.slot_labels:
                        self.slot_labels.append(slot)
        return self.intent_labels, self.slot_labels


    def get_data(self):

        # keep track of corrupt datapoints
        corrupt = 0

        for fp in self.file_paths:
            f = open( 'data/' + fp,'r')
            lines = f.readlines()
            # ------ check if train or test data is needed and set the list accordingly ------ #
            if self.type_req == 'train':
                print('Collecting train data ...')
            else:
                print('Collecting test data ...')
                lines.reverse()
            # ----- start loop to fill up data in the data structure ----- #
            i = 0
            for line in lines:
                if i == self.size:
                    break
                if i == len(lines):
                    print(f'WARNING: [{fp}] Requested size [{self.size}] exceeds file size [{self.size}]')
                else:
                    line = line.split('#')
                    # some sanity check to make sure data point is not corrupted:
                    # length must be 3 and first element should not be empty
                    try:
                        assert len(line) == 3 and len(line[0]) > 0
                        utterance = line[0]
                        intent = line[1]
                        slots = make_slot_dict(line[2])
                        # ---- make an instance of the corpus class ---- #
                        q = Query(utterance, intent, slots)
                        self.__corpus.append(q)
                        i+=1
                    except Exception:
                        corrupt += 1
                        print(f'WARNING: Corrupt data point: {line}')
        # DEBUG
        if corrupt:
            print(f'WARNING: {corrupt} corrupted data points were skipped!')
                    
                    

if __name__ == '__main__':

    corpus = Corpus(2000, 'train')
    #for x in corpus:
    #    print(x )
