from helper import *

class Corpus:
    def __init__(self, size, type_req):
        self.size = size
        self.type_req = type_req


    def get_train_data(self):
        
        data = list()
        files = ['slp_train_add_to_playlist.txt', 'slp_train_book_restaurant.txt', 'slp_train_get_weather.txt', 'slp_train_play_music.txt', 'slp_train_rate_book.txt', 'slp_train_search_creative_work.txt', 'slp_train_search_screening_event.txt']
        for each in files:
            f = open(each,'r') 
            lines = f.readlines()
            i = 0
            
            for line in lines:
                data_dict = dict()
                if i == self.size:
                    break
                else:
                    line = line.split('#')
                    slots = make_slot_dict(line[2])
                    data_dict['utterance'] = line[0]
                    data_dict['GoldIntent'] = line[1]
                    data_dict['PredictedIntent'] = None
                    data_dict['GoldSlots'] = slots
                    data_dict['PredictedSlots'] = None
                    data.append(data_dict)
                    i+=1

        return data

    def get_test_data(self):
        
        data = list()
        files = ['slp_train_add_to_playlist.txt', 'slp_train_book_restaurant.txt', 'slp_train_get_weather.txt', 'slp_train_play_music.txt', 'slp_train_rate_book.txt', 'slp_train_search_creative_work.txt', 'slp_train_search_screening_event.txt']
        for each in files:
            f = open(each,'r') 
            lines = f.readlines()
            i = 0
            
            for line in lines:
                data_dict = dict()
                if i == self.size:
                    break
                else:
                    line = line.split('#')
                    slots = make_slot_dict(line[2])
                    data_dict['utterance'] = line[0]
                    data_dict['GoldIntent'] = line[1]
                    data_dict['PredictedIntent'] = None
                    data_dict['GoldSlots'] = slots
                    data_dict['PredictedSlots'] = None
                    data.append(data_dict)
                    i+=1

        return data


    def operate(self):
        if self.type_req == 'train':
            return self.get_train_data()
        else:
            return self.get_test_data()



c = Corpus(10, 'train')

print(c.operate())
        