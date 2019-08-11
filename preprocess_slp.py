# --------- python script to preprocess data from the nlu-benchmark dataset ----------------------#

import json
# ------------------- Add to Playlist ------------------------- #
data = "/home/users0/sengupmt/Dokumente/nlu-benchmark-master/2017-06-custom-intent-engines/AddToPlaylist/train_AddToPlaylist.json"
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
            combined += entity + '=' + entity_text + '--'
    all_sent_playlist.append(to_combine + '#AddToPlaylist#' + combined)

f = open('slp_train_add_to_playlist.txt', 'w+')

for item in all_sent_playlist:
    f.write(item)
    f.write('\n')

# print(all_sent_playlist)

# ------------------- Book A restaurant ------------------------- #

data = "/home/users0/sengupmt/Dokumente/nlu-benchmark-master/2017-06-custom-intent-engines/BookRestaurant/train_BookRestaurant.json"
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
            combined += entity + '=' + entity_text + '--'
    all_sent_restaurant.append(to_combine + '#BookRestaurant#' + combined)

# -------------------- Write data to a file -------------------- #

f = open('slp_train_book_restaurant.txt', 'w+')

for item in all_sent_restaurant:
    f.write(item)
    f.write('\n')


# ------------------- Get Weather ------------------------- #

data = "/home/users0/sengupmt/Dokumente/nlu-benchmark-master/2017-06-custom-intent-engines/GetWeather/train_GetWeather.json"
with open(data, 'r') as f:
    data_to_process = f.read()

to_parse = json.loads(data_to_process)

parse_arr = to_parse['GetWeather']

all_sent_weather = list()

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
            combined += entity + '=' + entity_text + '--'
    all_sent_weather.append(to_combine + '#GetWeather#' + combined)

f = open('slp_train_get_weather.txt', 'w+')

for item in all_sent_weather:
    f.write(item)
    f.write('\n')
# ------------------- Play Music ------------------------- #

data = "/home/users0/sengupmt/Dokumente/nlu-benchmark-master/2017-06-custom-intent-engines/PlayMusic/train_PlayMusic.json"
with open(data, 'r') as f:
    data_to_process = f.read()

to_parse = json.loads(data_to_process)

parse_arr = to_parse['PlayMusic']

all_sent_play = list()

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
            combined += entity + '=' + entity_text + '--'
    all_sent_play.append(to_combine + '#PlayMusic#' + combined)

# -------------------- Write data to a file -------------------- #

f = open('slp_train_play_music.txt', 'w+')

for item in all_sent_play:
    f.write(item)
    f.write('\n')


# ------------------- Rate Book ------------------------- #

data = "/home/users0/sengupmt/Dokumente/nlu-benchmark-master/2017-06-custom-intent-engines/RateBook/train_RateBook.json"
with open(data, 'r') as f:
    data_to_process = f.read()

to_parse = json.loads(data_to_process)

parse_arr = to_parse['RateBook']

all_sent_rate = list()

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
            combined += entity + '=' + entity_text + '--'
    all_sent_rate.append(to_combine + '#RateBook#' + combined)

# -------------------- Write data to a file -------------------- #

f = open('slp_train_rate_book.txt', 'w+')

for item in all_sent_rate:
    f.write(item)
    f.write('\n')

# ------------------- Search Creative Work ------------------------- #

data = "/home/users0/sengupmt/Dokumente/nlu-benchmark-master/2017-06-custom-intent-engines/SearchCreativeWork/train_SearchCreativeWork.json"
with open(data, 'r') as f:
    data_to_process = f.read()

to_parse = json.loads(data_to_process)

parse_arr = to_parse['SearchCreativeWork']

all_sent_cr = list()

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
            combined += entity + '=' + entity_text + '--'
    all_sent_cr.append(to_combine + '#SearchCreativeWork#' + combined)

# -------------------- Write data to a file -------------------- #

f = open('slp_train_search_creative_work.txt', 'w+')

for item in all_sent_cr:
    f.write(item)
    f.write('\n')

# ------------------- Search Screening Event ------------------------- #

data = "/home/users0/sengupmt/Dokumente/nlu-benchmark-master/2017-06-custom-intent-engines/SearchScreeningEvent/train_SearchScreeningEvent.json"
with open(data, 'r') as f:
    data_to_process = f.read()

to_parse = json.loads(data_to_process)

parse_arr = to_parse['SearchScreeningEvent']

all_sent_scr = list()

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
            combined += entity + '=' + entity_text + '--'
    all_sent_scr.append(to_combine + '#SearchScreeningEvent#' + combined)

# -------------------- Write data to a file -------------------- #

f = open('slp_train_search_screening_event.txt', 'w+')

for item in all_sent_scr:
    f.write(item)
    f.write('\n')