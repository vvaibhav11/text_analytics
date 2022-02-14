# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:10:31 2022

@author: vvaib
"""

########################## Setup ######################################################
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pandas as pd
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import itertools
from collections import Counter
import statistics as st
import numpy as np
import collections
from wordcloud import WordCloud
import matplotlib.pyplot as plt

####################### Load data ################################
ac = pd.read_csv("AirCanada_review_raw.csv")
ac = ac.drop(ac.columns[0], axis=1)

##################################################################
# count of positive and negative reviews wrt seat types accross
# Value For Money
# Seat Comfort
# Cabin Staff Service
# Food & Beverages
# Inflight Entertainment
# Ground Service
# Wifi & Connectivity

a = pd.DataFrame(ac[ac['Value For Money'] > 0].groupby('Seat Type')['Value For Money'].count())
a['good_raters'] = (ac[ac['Value For Money'] > 2].groupby('Seat Type')['Value For Money'].count())
a['pos_share'] = a['good_raters']/a['Value For Money']*100
a['neg_share'] = 100 - a['pos_share']

b = pd.DataFrame(ac[ac['Seat Comfort'] > 0].groupby('Seat Type')['Seat Comfort'].count())
b['good_raters'] = (ac[ac['Seat Comfort'] > 2].groupby('Seat Type')['Seat Comfort'].count())
b['pos_share'] = b['good_raters']/b['Seat Comfort']*100
b['neg_share'] = 100 - b['pos_share']

c = pd.DataFrame(ac[ac['Cabin Staff Service'] > 0].groupby('Seat Type')['Cabin Staff Service'].count())
c['good_raters'] = (ac[ac['Cabin Staff Service'] > 2].groupby('Seat Type')['Cabin Staff Service'].count())
c['pos_share'] = c['good_raters']/c['Cabin Staff Service']*100
c['neg_share'] = 100 - c['pos_share']

d = pd.DataFrame(ac[ac['Food & Beverages'] > 0].groupby('Seat Type')['Food & Beverages'].count())
d['good_raters'] = (ac[ac['Food & Beverages'] > 2].groupby('Seat Type')['Food & Beverages'].count())
d['pos_share'] = d['good_raters']/d['Food & Beverages']*100
d['neg_share'] = 100 - d['pos_share']

e = pd.DataFrame(ac[ac['Inflight Entertainment'] > 0].groupby('Seat Type')['Inflight Entertainment'].count())
e['good_raters'] = (ac[ac['Inflight Entertainment'] > 2].groupby('Seat Type')['Inflight Entertainment'].count())
e['pos_share'] = e['good_raters']/e['Inflight Entertainment']*100
e['neg_share'] = 100 - e['pos_share']

f = pd.DataFrame(ac[ac['Ground Service'] > 0].groupby('Seat Type')['Ground Service'].count())
f['good_raters'] = (ac[ac['Ground Service'] > 2].groupby('Seat Type')['Ground Service'].count())
f['pos_share'] = f['good_raters']/f['Ground Service']*100
f['neg_share'] = 100 - e['pos_share']

g = pd.DataFrame(ac[ac['Wifi & Connectivity'] > 0].groupby('Seat Type')['Wifi & Connectivity'].count())
g['good_raters'] = (ac[ac['Wifi & Connectivity'] > 2].groupby('Seat Type')['Wifi & Connectivity'].count())
g['pos_share'] = g['good_raters']/g['Wifi & Connectivity']*100
g['neg_share'] = 100 - g['pos_share']

##################################################################
# count of positive and negative reviews wrt Type Of Traveller accross
# Value For Money
# Seat Comfort
# Cabin Staff Service
# Food & Beverages
# Inflight Entertainment
# Ground Service
# Wifi & Connectivity

a2 = pd.DataFrame(ac[ac['Value For Money'] > 0].groupby('Type Of Traveller')['Value For Money'].count())
a2['good_raters'] = (ac[ac['Value For Money'] > 2].groupby('Type Of Traveller')['Value For Money'].count())
a2['pos_share'] = a2['good_raters']/a2['Value For Money']*100
a2['neg_share'] = 100 - a2['pos_share']

b2 = pd.DataFrame(ac[ac['Seat Comfort'] > 0].groupby('Type Of Traveller')['Seat Comfort'].count())
b2['good_raters'] = (ac[ac['Seat Comfort'] > 2].groupby('Type Of Traveller')['Seat Comfort'].count())
b2['pos_share'] = b2['good_raters']/b2['Seat Comfort']*100
b2['neg_share'] = 100 - b2['pos_share']

c2 = pd.DataFrame(ac[ac['Cabin Staff Service'] > 0].groupby('Type Of Traveller')['Cabin Staff Service'].count())
c2['good_raters'] = (ac[ac['Cabin Staff Service'] > 2].groupby('Type Of Traveller')['Cabin Staff Service'].count())
c2['pos_share'] = c2['good_raters']/c2['Cabin Staff Service']*100
c2['neg_share'] = 100 - c2['pos_share']

d2 = pd.DataFrame(ac[ac['Food & Beverages'] > 0].groupby('Type Of Traveller')['Food & Beverages'].count())
d2['good_raters'] = (ac[ac['Food & Beverages'] > 2].groupby('Type Of Traveller')['Food & Beverages'].count())
d2['pos_share'] = d2['good_raters']/d2['Food & Beverages']*100
d2['neg_share'] = 100 - d2['pos_share']

e2 = pd.DataFrame(ac[ac['Inflight Entertainment'] > 0].groupby('Type Of Traveller')['Inflight Entertainment'].count())
e2['good_raters'] = (ac[ac['Inflight Entertainment'] > 2].groupby('Type Of Traveller')['Inflight Entertainment'].count())
e2['pos_share'] = e2['good_raters']/e2['Inflight Entertainment']*100
e2['neg_share'] = 100 - e2['pos_share']

f2 = pd.DataFrame(ac[ac['Ground Service'] > 0].groupby('Type Of Traveller')['Ground Service'].count())
f2['good_raters'] = (ac[ac['Ground Service'] > 2].groupby('Type Of Traveller')['Ground Service'].count())
f2['pos_share'] = f2['good_raters']/f2['Ground Service']*100
f2['neg_share'] = 100 - f2['pos_share']

g2 = pd.DataFrame(ac[ac['Wifi & Connectivity'] > 0].groupby('Type Of Traveller')['Wifi & Connectivity'].count())
g2['good_raters'] = (ac[ac['Wifi & Connectivity'] > 2].groupby('Type Of Traveller')['Wifi & Connectivity'].count())
g2['pos_share'] = g2['good_raters']/g2['Wifi & Connectivity']*100
g2['neg_share'] = 100 - g2['pos_share']

########################################################################################
############################ stop word removal #########################################

ac['no_stopword']= ac["review"].str.lower().tolist()

stop_words = stopwords.words('english')
english_stop_words = stop_words + ['air', 'canada', 'even', 'also', 'us', 'would', 'could', 'get', 'got', 'go', 'flight', 
                                   'flights', 'said', 'next', 'one', 'two', 'made', 'however', 'went', 'without', 'say', 'said',
                                   'Algiers',
                                    'Osbourn',
                                    'Buenos Aires',
                                    'Oranjestad',
                                    'Brisbane',
                                    'Melbourne',
                                    'Sydney',
                                    'Vienna',
                                    'George Town, Exuma',
                                    'Freeport',
                                    'Nassau',
                                    'San Salvador',
                                    'Bridgetown',
                                    'Brussels',
                                    'Hamilton',
                                    'Rio de Janeiro',
                                    'São Paulo',
                                    'Abbotsford',
                                    'Calgary',
                                    'Comox',
                                    'Deer Lake',
                                    'Edmonton',
                                    'Fort McMurray',
                                    'Halifax',
                                    'Hamilton, ON',
                                    'Kelowna',
                                    'London, ON',
                                    'Montréal',
                                    'Nanaimo',
                                    'Ottawa',
                                    'Quebec City',
                                    'Regina',
                                    'Saskatoon',
                                    'St.Johns',
                                    'Sydney, NS',
                                    'Toronto',
                                    'Vancouver',
                                    'Victoria',
                                    'Whitehorse',
                                    'Winnipeg',
                                    'Yellowknife',
                                    'George Town',
                                    'Santiago',
                                    'Beijing',
                                    'Shanghai',
                                    'Bogotá',
                                    'Liberia',
                                    'San José',
                                    'Zagreb',
                                    'Cayo Coco',
                                    'Cayo Largo del Sur',
                                    'Havana',
                                    'Holguín',
                                    'Santa Clara',
                                    'Varadero',
                                    'Curaçao',
                                    'Prague',
                                    'Copenhagen',
                                    'La Romana',
                                    'Puerto Plata',
                                    'Punta Cana',
                                    'Samana',
                                    'Santo Domingo',
                                    'Cairo',
                                    'Bordeaux',
                                    'Fort-de-France',
                                    'Lyon',
                                    'Nice',
                                    'Paris',
                                    'Pointe-à-Pitre',
                                    'Toulouse',
                                    'Berlin',
                                    'Düsseldorf',
                                    'Frankfurt',
                                    'Munich',
                                    'Athens',
                                    'St. Georges',
                                    'Port-au-Prince',
                                    'Hong Kong',
                                    'Budapest',
                                    'Reykjavík',
                                    'Delhi',
                                    'Mumbai',
                                    'Dublin',
                                    'Shannon',
                                    'Tel Aviv',
                                    'Milan',
                                    'Rome',
                                    'Venice',
                                    'Kingston',
                                    'Montego Bay',
                                    'Nagoya',
                                    'Osaka',
                                    'Tokyo',
                                    'Acapulco',
                                    'Bahías de Huatulco',
                                    'Cancún',
                                    'Cozumel',
                                    'Ixtapa/Zihuatanejo',
                                    'Mexico City',
                                    'Puerto Vallarta',
                                    'San José del Cabo',
                                    'Casablanca',
                                    'Amsterdam',
                                    'Auckland',
                                    'Panama City',
                                    'Lima',
                                    'Warsaw',
                                    'Lisbon',
                                    'Doha',
                                    'Moscow',
                                    'Basseterre',
                                    'Vieux-Fort',
                                    'St. Vincent',
                                    'Singapore',
                                    'Philipsburg',
                                    'Seoul',
                                    'Barcelona',
                                    'Madrid',
                                    'Geneva',
                                    'Zürich',
                                    'Taipei',
                                    'Port of Spain',
                                    'Istanbul',
                                    'Providenciales',
                                    'Dubai',
                                    'Belfast',
                                    'Birmingham',
                                    'Edinburgh',
                                    'Glasgow',
                                    'London',
                                    'Manchester',
                                    'Anchorage',
                                    'Austin',
                                    'Boston',
                                    'Charleston',
                                    'Charlotte',
                                    'Chicago',
                                    'Cleveland',
                                    'Columbia',
                                    'Dallas-Fort Worth',
                                    'Denver',
                                    'Fort Lauderdale',
                                    'Fort Myers',
                                    'Honolulu',
                                    'Houston',
                                    'Jacksonville',
                                    'Kahului',
                                    'Kona',
                                    'Las Vegas',
                                    'Lihue',
                                    'Los Angeles',
                                    'Memphis',
                                    'Miami',
                                    'Minneapolis–Saint Paul',
                                    'Myrtle Beach',
                                    'Newark',
                                    'New York City',
                                    'Ontario',
                                    'Orange County',
                                    'Orlando',
                                    'Palm Springs',
                                    'Philadelphia',
                                    'Phoenix',
                                    'Portland',
                                    'Richmond',
                                    'Salt Lake City',
                                    'Sacramento',
                                    'San Diego',
                                    'San Francisco',
                                    'San Jose, CA',
                                    'San Juan',
                                    'Seattle',
                                    'Tampa',
                                    'Vail',
                                    'Washington, D.C.',
                                    'West Palm Beach',
                                    'Caracas',
                                    'Porlamar',
                                    'City',
                                    'Halifax',
                                    'St. Johns',
                                    'Toronto',
                                    'Quito',
                                    'Frankfurt',
                                    'Guadalajara',
                                    'Mexico City',
                                    'Lima',
                                    'Madrid',
                                    'Miami',
                                    'City',
                                    'Algiers',
                                    'San Salvador',
                                    'Bridgetown',
                                    'Belize City',
                                    'Abbotsford',
                                    'Calgary',
                                    'Charlottetown',
                                    'Deer Lake',
                                    'Edmonton',
                                    'Fredericton',
                                    'Halifax',
                                    'Hamilton',
                                    'Kamloops',
                                    'Kelowna',
                                    'Montréal',
                                    'Moncton',
                                    'Nanaimo',
                                    'Ottawa',
                                    'Quebec City',
                                    'Regina',
                                    'Saint John',
                                    'St. Johns',
                                    'Sydney',
                                    'Thunder Bay',
                                    'Toronto',
                                    'Vancouver',
                                    'Victoria',
                                    'Bogota',
                                    'Cartagena',
                                    'Liberia',
                                    'San José',
                                    'Zagreb',
                                    'Cayo Coco',
                                    'Cayo Largo del Sur',
                                    'Havana',
                                    'Holguín',
                                    'Santa Clara',
                                    'Varadero',
                                    'Curaçao',
                                    'Prague',
                                    'La Romana',
                                    'Puerto Plata',
                                    'Punta Cana',
                                    'Samana',
                                    'Quito',
                                    'Bordeaux',
                                    'Marseille',
                                    'Nice',
                                    'Berlin',
                                    'Athens',
                                    'St. Georges',
                                    'Port-au-Prince',
                                    'Budapest',
                                    'Reykjavík',
                                    'Rome',
                                    'Venice',
                                    'Kingston',
                                    'Montego Bay',
                                    'Nagoya',
                                    'Osaka',
                                    'Cozumel',
                                    'Huatulco',
                                    'Ixtapa',
                                    'Mexico City',
                                    'Puerto Vallarta',
                                    'San José del Cabo',
                                    'Casablanca',
                                    'Panama City',
                                    'Lima',
                                    'Warsaw',
                                    'Lisbon',
                                    'Porto',
                                    'Bucharest',
                                    'Basseterre',
                                    'Philipsburg',
                                    'Barcelona',
                                    'Vieux Fort',
                                    'St. Vincent',
                                    'Port of Spain',
                                    'Providenciales',
                                    'Edinburgh',
                                    'Glasgow',
                                    'London',
                                    'Manchester',
                                    'Anchorage',
                                    'Fort Lauderdale',
                                    'Fort Myers',
                                    'Honolulu',
                                    'Kahului',
                                    'Kona',
                                    'Las Vegas',
                                    'Los Angeles',
                                    'Miami',
                                    'Orlando',
                                    'Palm Springs',
                                    'Phoenix',
                                    'Portland',
                                    'San Diego',
                                    'San Francisco',
                                    'Sarasota',
                                    'Tampa',
                                    'West Palm Beach',
                                    'City',
                                    'Abbotsford',
                                    'Bagotville',
                                    'Baie-Comeau',
                                    'Bathurst',
                                    'Calgary',
                                    'Campbell River',
                                    'Castlegar',
                                    'Charlottetown',
                                    'Comox',
                                    'Cranbrook',
                                    'Deer Lake',
                                    'Edmonton',
                                    'Fort McMurray',
                                    'Fort St. John',
                                    'Fredericton',
                                    'Gander',
                                    'Gaspé',
                                    'Grande Prairie',
                                    'Halifax',
                                    'Hamilton',
                                    'Happy Valley-Goose Bay',
                                    'Îles de la Madeleine',
                                    'Iqaluit',
                                    'Kamloops',
                                    'Kelowna',
                                    'Kingston',
                                    'Lethbridge',
                                    'London',
                                    'Medicine Hat',
                                    'Moncton',
                                    'Mont-Joli',
                                    'Mont-Tremblant',
                                    'Montreal',
                                    'Nanaimo',
                                    'North Bay',
                                    'Ottawa',
                                    'Penticton',
                                    'Prince George',
                                    'Prince Rupert',
                                    'Quebec City',
                                    'Quesnel',
                                    'Red Deer',
                                    'Regina',
                                    'Rouyn-Noranda',
                                    'Saint John',
                                    'Saint-Léonard',
                                    'Sandspit',
                                    'Sarnia',
                                    'Saskatoon',
                                    'Sault Ste. Marie',
                                    'Sept-Îles',
                                    'Smithers',
                                    'Stephenville',
                                    'St. Johns',
                                    'Sudbury',
                                    'Sydney',
                                    'Terrace',
                                    'Thunder Bay',
                                    'Timmins',
                                    'Toronto',
                                    'Val-dOr',
                                    'Vancouver',
                                    'Victoria',
                                    'Wabush',
                                    'Whitehorse',
                                    'Williams Lake',
                                    'Windsor',
                                    'Winnipeg',
                                    'Yarmouth',
                                    'Yellowknife',
                                    'Albany',
                                    'Allentown',
                                    'Atlanta',
                                    'Atlantic City',
                                    'Austin',
                                    'Baltimore',
                                    'Boston',
                                    'Charlotte',
                                    'Chicago',
                                    'Cincinnati/Covington',
                                    'Cleveland',
                                    'Columbus',
                                    'Dallas/Fort Worth',
                                    'Dayton',
                                    'Denver',
                                    'Detroit',
                                    'Grand Rapids',
                                    'Harrisburg',
                                    'Hartford',
                                    'Houston',
                                    'Indianapolis',
                                    'Jacksonville',
                                    'Kansas City',
                                    'Las Vegas',
                                    'Los Angeles',
                                    'Manchester',
                                    'Memphis',
                                    'Milwaukee',
                                    'Minneapolis–Saint Paul',
                                    'Nashville',
                                    'Newark',
                                    'New Orleans',
                                    'New York City',
                                    'Norfolk',
                                    'Omaha',
                                    'Palm Springs',
                                    'Philadelphia',
                                    'Phoenix',
                                    'Pittsburgh',
                                    'Portland (ME)',
                                    'Portland (OR)',
                                    'Providence',
                                    'Raleigh/Durham',
                                    'Richmond',
                                    'Rochester',
                                    'Sacramento',
                                    'San Antonio',
                                    'San Diego',
                                    'San Jose',
                                    'San Francisco',
                                    'Savannah',
                                    'Seattle/Tacoma',
                                    'St. Louis',
                                    'Syracuse',
                                    'Washington D.C.',
                                    'White Plains']

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words and word.isalpha()])
        )
    return removed_stop_words

ac['no_stopword'] = remove_stop_words(ac['no_stopword'])

################################# Limmization ##########################################

def get_lemmatized_text(corpus):
    
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

ac['limmizated'] = get_lemmatized_text(ac['no_stopword'])

#################### Tokenize lemmatized comments #####################################

def apwords(words):
    tokenize = []
    words = nltk.pos_tag(word_tokenize(words))
    for w in words:
        tokenize.append(w)
    return tokenize

addwords = lambda x: apwords(x)
ac['tokensize'] = ac['limmizated'].apply(addwords)

#################### removing unwarrented tags ########################################

token = ac['tokensize'].tolist()

final_tags = [ [] for i in range(1644) ]

tags = ['FW','JJ','JJR','JJS','MD','NN','NNS','NNP','NNPS','RB','RBR','RBS',
        'SYM','VB','VBD','VBG','VBN']

for i in range(len(token)):
    #print(i)
    for j in range(len(token[i])):
        #print(j)
        for k in tags:
            if k == token[i][j][1]:
                final_tags[i].append(token[i][j][0])
                continue

ac['filtered_tags'] = final_tags


##################### Frequency of words (+ve/-ve) for Value for Money with respect to seat types #######################
# Positive words
f1 = pd.DataFrame(ac[(ac['Value For Money'] > 2)])
c1 = list(itertools.chain.from_iterable(f1['filtered_tags']))

counts = Counter(c1)
data_items = counts.items()
data_list = list(data_items)

v_money_wf_pos = pd.DataFrame(data_list)
v_money_wf_pos.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
v_money_wf_pos = v_money_wf_pos.sort_values(by = 'frequency', ascending = False)

data = v_money_wf_pos.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Negative words
f2 = pd.DataFrame(ac[(ac['Value For Money'] > 2)])
c2 = list(itertools.chain.from_iterable(f2['filtered_tags']))

counts = Counter(c2)
data_items = counts.items()
data_list = list(data_items)

v_money_neg = pd.DataFrame(data_list)
v_money_neg.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
v_money_neg = v_money_neg.sort_values(by = 'frequency', ascending = False)

data = v_money_neg.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

##################### Frequency of words (+ve/-ve) for Seat comfort with respect to seat types #######################
# Positive words
f3 = pd.DataFrame(ac[(ac['Seat Comfort'] > 2)])
c3 = list(itertools.chain.from_iterable(f3['filtered_tags']))

counts = Counter(c3)
data_items = counts.items()
data_list = list(data_items)

seat_wf_pos = pd.DataFrame(data_list)
seat_wf_pos.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
seat_wf_pos = seat_wf_pos.sort_values(by = 'frequency', ascending = False)

data = seat_wf_pos.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Negative words
f4 = pd.DataFrame(ac[(ac['Seat Comfort'] > 2)])
c4 = list(itertools.chain.from_iterable(f4['filtered_tags']))

counts = Counter(c4)
data_items = counts.items()
data_list = list(data_items)

seat_wf_neg = pd.DataFrame(data_list)
seat_wf_neg.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
seat_wf_neg = seat_wf_neg.sort_values(by = 'frequency', ascending = False)

data = seat_wf_neg.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

##################### Frequency of words (+ve/-ve) for Cabin Staff Service with respect to seat types #######################
# Positive words
f5 = pd.DataFrame(ac[(ac['Cabin Staff Service'] > 2)])
c5 = list(itertools.chain.from_iterable(f5['filtered_tags']))

counts = Counter(c5)
data_items = counts.items()
data_list = list(data_items)

crew_wf_pos = pd.DataFrame(data_list)
crew_wf_pos.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
crew_wf_pos = crew_wf_pos.sort_values(by = 'frequency', ascending = False)

data = crew_wf_pos.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Negative words
f6 = pd.DataFrame(ac[(ac['Cabin Staff Service'] > 2)])
c6 = list(itertools.chain.from_iterable(f6['filtered_tags']))

counts = Counter(c6)
data_items = counts.items()
data_list = list(data_items)

crew_wf_neg = pd.DataFrame(data_list)
crew_wf_neg.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
crew_wf_neg = crew_wf_neg.sort_values(by = 'frequency', ascending = False)

data = crew_wf_neg.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

##################### Frequency of words (+ve/-ve) for Food & Beverages with respect to seat types #######################
# Positive words
f7 = pd.DataFrame(ac[(ac['Food & Beverages'] > 2)])
c7 = list(itertools.chain.from_iterable(f7['filtered_tags']))

counts = Counter(c7)
data_items = counts.items()
data_list = list(data_items)

food_wf_pos = pd.DataFrame(data_list)
food_wf_pos.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
food_wf_pos = food_wf_pos.sort_values(by = 'frequency', ascending = False)

data = food_wf_pos.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Negative words
f8 = pd.DataFrame(ac[(ac['Food & Beverages'] > 2)])
c8 = list(itertools.chain.from_iterable(f8['filtered_tags']))

counts = Counter(c8)
data_items = counts.items()
data_list = list(data_items)

food_wf_neg = pd.DataFrame(data_list)
food_wf_neg.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
food_wf_neg = food_wf_neg.sort_values(by = 'frequency', ascending = False)

data = food_wf_neg.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

##################### Frequency of words (+ve/-ve) for Inflight Entertainment with respect to seat types #######################
# Positive words
f9 = pd.DataFrame(ac[(ac['Inflight Entertainment'] > 2)])
c9 = list(itertools.chain.from_iterable(f9['filtered_tags']))

counts = Counter(c9)
data_items = counts.items()
data_list = list(data_items)

inflight_wf_pos = pd.DataFrame(data_list)
inflight_wf_pos.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
inflight_wf_pos = inflight_wf_pos.sort_values(by = 'frequency', ascending = False)

data = inflight_wf_pos.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Negative words
f10 = pd.DataFrame(ac[(ac['Inflight Entertainment'] > 2)])
c10 = list(itertools.chain.from_iterable(f10['filtered_tags']))

counts = Counter(c10)
data_items = counts.items()
data_list = list(data_items)

inflight_wf_neg = pd.DataFrame(data_list)
inflight_wf_neg.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
inflight_wf_neg = inflight_wf_neg.sort_values(by = 'frequency', ascending = False)

data = inflight_wf_neg.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

##################### Frequency of words (+ve/-ve) for Ground Service with respect to seat types #######################
# Positive words
f11 = pd.DataFrame(ac[(ac['Ground Service'] > 2)])
c11 = list(itertools.chain.from_iterable(f11['filtered_tags']))

counts = Counter(c11)
data_items = counts.items()
data_list = list(data_items)

ground_wf_pos = pd.DataFrame(data_list)
ground_wf_pos.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
ground_wf_pos = ground_wf_pos.sort_values(by = 'frequency', ascending = False)

data = ground_wf_pos.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Negative words
f12 = pd.DataFrame(ac[(ac['Ground Service'] > 2)])
c12 = list(itertools.chain.from_iterable(f12['filtered_tags']))

counts = Counter(c12)
data_items = counts.items()
data_list = list(data_items)

ground_wf_neg = pd.DataFrame(data_list)
ground_wf_neg.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
ground_wf_neg = seat_wf_pos.sort_values(by = 'frequency', ascending = False)

data = ground_wf_neg.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

##################### Frequency of words (+ve/-ve) for Wifi & Connectivity with respect to seat types #######################
# Positive words
f13 = pd.DataFrame(ac[(ac['Wifi & Connectivity'] > 2)])
c13 = list(itertools.chain.from_iterable(f13['filtered_tags']))

counts = Counter(c13)
data_items = counts.items()
data_list = list(data_items)

wifi_wf_pos = pd.DataFrame(data_list)
wifi_wf_pos.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
wifi_wf_pos = wifi_wf_pos.sort_values(by = 'frequency', ascending = False)

data = wifi_wf_pos.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Negative words
f14 = pd.DataFrame(ac[(ac['Wifi & Connectivity'] > 2)])
c14 = list(itertools.chain.from_iterable(f14['filtered_tags']))

counts = Counter(c14)
data_items = counts.items()
data_list = list(data_items)

wifi_wf_neg = pd.DataFrame(data_list)
wifi_wf_neg.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
wifi_wf_neg = wifi_wf_neg.sort_values(by = 'frequency', ascending = False)

data = wifi_wf_neg.set_index('word').to_dict()['frequency']
wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
