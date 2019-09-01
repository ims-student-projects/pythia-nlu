from helper import *
from query import *

class Corpus:
    def __init__(self, size, type_req):
        self.size = size
        self.type_req = type_req
        self.__corpus = list()
        self.__curr = 0 # counter for iterator


    def __next__(self):
        if self.__curr >= self.length():
            raise StopIteration
        else:
            self.__curr += 1
            return self.__corpus[self.__curr - 1]


    def __iter__(self):
        return iter(self.__corpus)


    def get_data(self):
        
        data = list()

        files = ['slp_train_add_to_playlist.txt', 'slp_train_book_restaurant.txt', 'slp_train_get_weather.txt', 'slp_train_play_music.txt', 'slp_train_rate_book.txt', 'slp_train_search_creative_work.txt', 'slp_train_search_screening_event.txt']
        
        for each in files:
            f = open(each,'r') 
            lines = f.readlines()
            
            
            # ------ check if train or test data is needed and set the list accordingly ------ #
            if self.type_req == 'train':
                print('Collecting train data ...')
            else:
                print('Collecting test data ...')
                lines = reversed(lines)

            
            # ----- start loop to fill up data in the data structure ----- #
            i = 0
            for line in lines:
                data_dict = dict()
                if i == self.size:
                    break
                else:
                    line = line.split('#')
                    utterance = line[0]
                    intent = line[1]
                    slots = make_slot_dict(line[2])
                    # ---- make an instance of the corpus class ---- #
                    q = Query(utterance, intent, slots)
                    self.__corpus.append(q)
                    i+=1    

    # def operate(self):
    #     return self.get_data()

# c = Corpus(2, 'train')

# c.operate()

# print(c.operate())
        

