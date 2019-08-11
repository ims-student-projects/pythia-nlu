# --------- python script to preprocess data from the nlu-benchmark dataset ----------------------#

import json
# ------------------- Add to Playlist ------------------------- #
data = "/Users/user/Documents/nlu-benchmark-master/2017-06-custom-intent-engines/AddToPlaylist/train_AddToPlaylist.json"
with open(data, 'r') as f:
    data_to_process = f.read()

to_parse = json.loads(data_to_process)

parse_arr = to_parse['AddToPlaylist']

all_sent_playlist = list()

for i in range(len(parse_arr)):
    inner_ls = parse_arr[i]['data']
    
    # combine sentences to write to file

    to_combine = ''
    combined = ''
    for j in range(len(inner_ls)):
        to_combine += inner_ls[j]['text']
        if len(inner_ls[j])>1:
            entity = inner_ls[j]['entity']
            entity_text = inner_ls[j]['text']
            combined += entity + ' = ' + entity_text + ', '
    all_sent_playlist.append(to_combine + ', AddToPlaylist, ' + combined)

# print(all_sent_playlist)

# ------------------- Book A restaurant ------------------------- #

data = "/Users/user/Documents/nlu-benchmark-master/2017-06-custom-intent-engines/BookRestaurant/train_BookRestaurant.json"
with open(data, 'r') as f:
    data_to_process = f.read()

to_parse = json.loads(data_to_process)

parse_arr = to_parse['BookRestaurant']

all_sent_restaurant = list()

for i in range(len(parse_arr)):
    inner_ls = parse_arr[i]['data']
    
    # combine sentences to write to file

    to_combine = ''
    combined = ''
    for j in range(len(inner_ls)):
        to_combine += inner_ls[j]['text']
        if len(inner_ls[j])>1:
            entity = inner_ls[j]['entity']
            entity_text = inner_ls[j]['text']
            combined += entity + ' = ' + entity_text + ', '
    all_sent_restaurant.append(to_combine + ', BookRestaurant, ' + combined)

# -------------------- Write data to a file -------------------- #

alles = all_sent_playlist + all_sent_restaurant

f = open('slp_train.txt', 'w+')

for item in alles:
    f.write(item)
    f.write('\n')
