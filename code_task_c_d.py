# -*- coding: utf-8 -*-
"""
Created on Mon Jan  31 22:41:04 2022

@author: vvaib
"""

########################## Setup #############################################
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

############## Loading data files (comments and brand records) ###############

c1 = pd.read_csv("comments_raw.csv")
c1 = c1.head(5000)
m1 = pd.read_csv("models.csv", header = None)
m1.rename(columns={0: 'brand', 1: 'model'}, inplace=True)
m1['brand'] = m1['brand'].str.lower()
m1['model'] = m1['model'].str.lower()

############################ stop word removal ################################

c1['no_stopword']= c1["comment"].str.lower().tolist()

english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words and word.isalpha()])
        )
    return removed_stop_words

c1['no_stopword'] = remove_stop_words(c1['no_stopword'])

################################# Limmization #################################

def get_lemmatized_text(corpus):
    
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

c1['limmizated'] = get_lemmatized_text(c1['no_stopword'])

#################### Tokenize lemmatized comments #############################

def apwords(words):
    tokenize = []
    words = nltk.pos_tag(word_tokenize(words))
    for w in words:
        tokenize.append(w)
    return tokenize

addwords = lambda x: apwords(x)
c1['tokensize'] = c1['limmizated'].apply(addwords)

#################### removing unwarrented tags ################################

c2 = c1['tokensize'].tolist()

final_tags = [ [] for i in range(5000) ]
tags = ['FW','JJ','JJR','JJS','MD','NN','NNS','NNP','NNPS','RB','RBR','RBS',
        'SYM','VB','VBD','VBG','VBN']

for i in range(len(c2)):
    #print(i)
    for j in range(len(c2[i])):
        #print(j)
        for k in tags:
            if k == c2[i][j][1]:
                final_tags[i].append(c2[i][j][0])
                continue

c1['filtered_tags'] = final_tags

##################### Frequency count of words of cleaned reviews #############

c3 = list(itertools.chain.from_iterable(c1['filtered_tags']))

counts = Counter(c3)
data_items = counts.items()
data_list = list(data_items)

word_frequency = pd.DataFrame(data_list)
word_frequency.rename(columns = {0:'word', 1:'frequency'}, inplace = True)
word_frequency = word_frequency.sort_values(by = 'frequency', ascending = False)

#################### Count of Brand and model occurance ################################# 

brand_list = []
model_list = []

for w in word_frequency['word']:
    for c in m1.brand.unique():
        if w == c:
            brand_list.append(w)
    for m in m1.model.unique():
        if w == m:
            model_list.append(w)
        

####### Changing car names with their model names in comments using brand data set #######

word_frequency2 = word_frequency.copy()
for m in model_list:
    mcar = m1.loc[m1['model'] == m, 'brand'].iloc[0]
    word_frequency2 = word_frequency2.replace(m, mcar)

word_frequency2 = word_frequency2.groupby(['word'])['frequency'].apply(np.sum).reset_index()
word_frequency2 = word_frequency2.sort_values(by = 'frequency', ascending = False)

###### Filtering frequency based on distribution (top words with 60% of total counts) ######

print(sum(word_frequency['frequency']))
print(sum(word_frequency2['frequency']))

print(word_frequency2['frequency'].quantile(0.97))
word_frequency2 = word_frequency2[word_frequency2['frequency'] > word_frequency2['frequency'].quantile(0.97)] 
print(sum(word_frequency2['frequency']))

############ breaking cleaned taged comments from list to string ##############

c1['filtered_tags'] = [','.join(map(str, l)) for l in c1['filtered_tags']]
c1['filtered_tags'] = c1['filtered_tags'].str.replace(',', ' ')

#################### Top 5 brands ####################

top_brands = []

for w in word_frequency2['word']:
    for c in m1.brand.unique():
        if w == c:
            top_brands.append(w)

top_brands = pd.DataFrame(top_brands).head(5)
top_brands.rename(columns={0: 'brand'}, inplace=True)

###################### Attributes and brand flags #######################
# Brand flags
final = c1

final['honda'] = np.where(pd.DataFrame(final['filtered_tags'].str.contains('honda')),1,0)
final['ford'] = np.where(pd.DataFrame(final['filtered_tags'].str.contains('ford')),1,0)
final['toyota'] = np.where(pd.DataFrame(final['filtered_tags'].str.contains('toyota')),1,0)
final['hyundai'] = np.where(pd.DataFrame(final['filtered_tags'].str.contains('hyundai')),1,0)
final['nissan'] = np.where(pd.DataFrame(final['filtered_tags'].str.contains('nissan')),1,0)

# Attribute flags
condition = ['new','used','year','good','better','quality','reliability']
performance = ['engine','drive','power','hp','manual','speed','transmission','performance','handling','cylinder','mpg',
               'automatic','highway','torque','sport','steering']
fuel_efficiency = ['mpg','mile','fuel','mileage','gas']
value_money = ['price','cost','sale','value','warrenty','sell','resale']
looks = ['pretty','interior','looking','wheel','nice','seat','design','tire']

pattern1 = '|'.join(condition)
pattern2 = '|'.join(performance)
pattern3 = '|'.join(fuel_efficiency)
pattern4 = '|'.join(value_money)
pattern5 = '|'.join(looks)

final['condition'] = np.where(pd.DataFrame(final['filtered_tags'].str.contains(pattern1)),1,0)
final['performance'] = np.where(pd.DataFrame(final['filtered_tags'].str.contains(pattern2)),1,0)
final['fuel_efficiency'] = np.where(pd.DataFrame(final['filtered_tags'].str.contains(pattern3)),1,0)
final['value_money'] = np.where(pd.DataFrame(final['filtered_tags'].str.contains(pattern4)),1,0)
final['looks'] = np.where(pd.DataFrame(final['filtered_tags'].str.contains(pattern5)),1,0)

##### Writing the final cleaned comments with flags for top 5 brands and attributes #####
final.to_csv('comments_v2.csv', index=False, na_rep='Unknown')

################## Attributes analysis on top 5 brands #######################

honda = final.groupby(['honda'])["condition", "performance","fuel_efficiency","value_money","looks"].apply(lambda x : x.astype(int).sum())
ford = final.groupby(['ford'])["condition", "performance","fuel_efficiency","value_money","looks"].apply(lambda x : x.astype(int).sum())
toyota = final.groupby(['toyota'])["condition", "performance","fuel_efficiency","value_money","looks"].apply(lambda x : x.astype(int).sum())
hyundai = final.groupby(['hyundai'])["condition", "performance","fuel_efficiency","value_money","looks"].apply(lambda x : x.astype(int).sum())
nissan = final.groupby(['nissan'])["condition", "performance","fuel_efficiency","value_money","looks"].apply(lambda x : x.astype(int).sum())
