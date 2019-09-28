import sys
import random
import json

sys.path.append(sys.path[0] + '/../')
from corpus.query import Query

class Corpus:
    def __init__(self, size, type_req):
        # private attributes
        self.__corpus = list()
        self.__curr = 0 # counter for iterator
        # public attributes
        self.size = size
        self.total_size = 0
        self.type_req = type_req
        self.intent_labels = {}
        self.slot_labels = []
        self.train_files = ['train_SearchCreativeWork_full.json',
                            'train_PlayMusic_full.json',
                            'train_SearchScreeningEvent_full.json',
                            'train_GetWeather_full.json',
                            'train_AddToPlaylist_full.json',
                            'train_BookRestaurant_full.json',
                            'train_RateBook_full.json']
        self.test_files = [ 'validate_RateBook.json',
                            'validate_AddToPlaylist.json',
                            'validate_GetWeather.json',
                            'validate_PlayMusic.json',
                            'validate_SearchCreativeWork.json',
                            'validate_BookRestaurant.json',
                            'validate_SearchScreeningEvent.json']
        # import data from files
        self.get_data(file_paths=self.train_files if type_req == 'train' else self.test_files)


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

    def get_sample_by_index(self, index):
        return self.__corpus[index]

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


    def get_data(self, file_paths):

        # keep track of corrupt datapoints
        corrupt = 0

        for fp in file_paths:
            with open ( 'data/' + fp, encoding='ISO-8859-1') as raw_data:
                data = json.load(raw_data)
                intent = list(data.keys())[0]

                # if requested size exceeds size, only return what is there
                n = self.size if self.size <= len(data[intent]) else len(data[intent])

                for x in data[intent][:n]:
                    try:
                        utterance = ''.join(slot['text'] for slot in x['data'])
                        slots = {slot['entity']: slot['text'] for slot in x['data'] if 'entity' in slot}
                        q = Query(utterance, intent, slots)
                        self.__corpus.append(q)
                        self.total_size += 1
                    except Exception as ex:
                        corrupt += 1
                        print(f'ERROR: Unable to parse ({ex}): [{x}]')
        # DEBUG
        if corrupt:
            print(f'WARNING: {corrupt} corrupted data points were skipped.')

                    
    def get_size(self):
        return self.total_size
                          

if __name__ == '__main__':

    corpus = Corpus(2000, 'train')
    #for x in corpus:
    #    print(x )
