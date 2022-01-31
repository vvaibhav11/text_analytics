# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 14:29:19 2022

@author: vvaib
"""
###################### Setup #######################################
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pandas as pd
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

comments = pd.read_csv("comments_raw.csv")
c2 = comments[['comment']]
c2 = c2["comment"].tolist()

#####################################################################
###################### Remmoving stop words #########################

english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

c2_nostop = remove_stop_words(c2)


#####################################################################
##################### Stemming of reviews ########################### 

def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

c2_stemmed = get_stemmed_text(c2)


#####################################################################
##################### Lemmatization of reviews ######################

def get_lemmatized_text(corpus):
    
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

c2_lemmatized = get_lemmatized_text(c2)

#####################################################################
##################### word counts analysis ##########################

c2_lemmatized2 = pd.DataFrame (c2_lemmatized, columns = ['comment'])

# cleaning review column
c2_lemmatized2 = c2_lemmatized2[~c2_lemmatized2['comment'].isnull()]

def preprocess(commenttext):
    commenttext = commenttext.str.replace("(<br/>)", "")
    commenttext = commenttext.str.replace('(<a).*(>).*(</a>)', '')
    commenttext = commenttext.str.replace('(&amp)', '')
    commenttext = commenttext.str.replace('(&gt)', '')
    commenttext = commenttext.str.replace('(&lt)', '')
    commenttext = commenttext.str.replace('(\xa0)', ' ')  
    return commenttext

c2_lemmatized2['comment'] = preprocess(c2_lemmatized2['comment'])

# word count, character length and word sequences
c2_lemmatized2['polarity'] = c2_lemmatized2['comment'].map(lambda text: TextBlob(text).sentiment.polarity)
c2_lemmatized2['comment_len'] = c2_lemmatized2['comment'].astype(str).apply(len)
c2_lemmatized2['word_count'] = c2_lemmatized2['comment'].apply(lambda x: len(str(x).split()))

#####################################################################
##################### TF-IDF analysis ###############################


#####################################################################
##################### n gram analysis ###############################

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
c2_ngram = ngram_vectorizer.fit(c2)
c2_ngram = ngram_vectorizer.transform(c2)
print(c2_ngram)

#####################################################################
# writing comment data with no stop words, stemming and lemmatizing #

comments['comment_nostop'] = c2_nostop
comments['comment_stemming'] = c2_nostop
comments['comment_lemmatize'] = c2_nostop
comments['polarity_lemmit'] = c2_lemmatized2['polarity']
comments['word_count_lemmit'] = c2_lemmatized2['word_count']
comments['comment_len_lemmit'] = c2_lemmatized2['comment_len']

comments.to_csv('comments_v2.csv', index=False, na_rep='Unknown')


