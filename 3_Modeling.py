# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:36:34 2022

@author: Barrett
"""

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re


# Get all the tweets.
path = r'D:\Springboard_DataSci\Twitter_MBTI_predictor\Data Output'
os.chdir(path)

TYPE = 'Type'
LOGISTIC = 'logistic'
NAIVE_BAYES = 'multinomial naive Bayes'

letters = [['E', 'I'], ['S', 'N'], ['F', 'T'], ['J', 'P']]
MB_types = []
# Get the list of types using binary math.
for i in range(16):
    MB_types.append(letters[0][i//8%2] + letters[1][i//4%2]
                      + letters[2][i//2%2] + letters[3][i%2])
    
def load_tweets(MB_type):
    return pd.read_csv(
        path + '\\' + MB_type + '_tweets.csv', parse_dates=[2],
        infer_datetime_format=True)

# Load tweets
print('Loading tweets:', end=' ')
for i, MB_type in enumerate(MB_types):
    print(f'{MB_type}', end=' ')
    if i == 0:
        tweets = load_tweets(MB_type)
    else:
        tweets = tweets.append(load_tweets(MB_type))
        
# Classify their type
for i, letter in enumerate('ESFJ'):
    tweets[letter] = tweets['MBTI'].str[i] == letter
        
#%% Trim tweets
'''We're going to pick up a lot of junk if we don't trim out tags and hashtags.
Let's do that now.'''
def trim_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)",
                           " ", tweet).split())

print('Trimming tweets of tags and URLs')
tweets['Tweet'] = tweets['Tweet'].apply(trim_tweet)
        
#%% Analyze tweets; sort words by score
'''Now let's attempt to classify.'''
def analyze_tweets(tweets, letter, classifier, min_df=200, max_df=1.,
                   alpha=1., C=1, stop_words=None, get_words_and_probas=False,
                   test_size=0.25, max_iter=1e3):
    '''Text classification of the tweets'''
    y = tweets[letter]
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df,
                                 stop_words=stop_words)
    tweets = tweets['Tweet'].to_list()
    # Get the sparse matrix (x, y) of (tweetID, wordID).
    X = vectorizer.fit_transform(tweets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0)
    if classifier==LOGISTIC:
        clf = LogisticRegression(C=C, max_iter=max_iter, random_state=0)\
            .fit(X_train, y_train)
    if classifier==NAIVE_BAYES:
        clf = MultinomialNB(alpha=alpha).fit(X_train, y_train)
    if get_words_and_probas:
        x = np.eye(X_test.shape[1])
        words_all = np.array(vectorizer.get_feature_names())
        probs = clf.predict_log_proba(x)[:, 0]
    else:
        words_all = probs = None
    return clf.score(X_train, y_train), clf.score(X_test, y_test), words_all,\
        probs

#%% Grouping by author instead of by tweet
print('Grouping tweets by author')
tweets_per_author = tweets.copy()
tweets_per_author['Tweet'] = tweets_per_author['Tweet']\
    .apply(lambda x: x + ' ')
tweets_per_author = tweets_per_author.groupby(
    tweets_per_author['Screen name'])['Tweet'].apply(lambda x: x.sum())\
    .reset_index()
# This threw away the MBTI info, but we can get it back.
authors_MBTI = tweets[['Screen name', 'E', 'S', 'F', 'J']].drop_duplicates()
tweets_per_author = tweets_per_author.merge(
    authors_MBTI, 'left', on='Screen name')

#%%
print('\nAnalyzing tweets at the tweet level')
min_df=500
tweets_results = analyze_tweets(tweets, 'E', classifier=NAIVE_BAYES,
                                min_df=min_df, get_words_and_probas=True)
print(f'Full set, min_df={min_df} training and test scores:', end=' ')
print(round(tweets_results[0], 4), round(tweets_results[1], 4))

#%%
test_size = 0.2
print(f'\nAnalyzing tweets at the author level, test_size={test_size}')
min_df=500
author_results = analyze_tweets(
    tweets_per_author, 'E', classifier=NAIVE_BAYES, min_df=min_df,
    get_words_and_probas=True, test_size=test_size)
print(f'Full set, min_df={min_df} training and test scores:', end=' ')
print(round(author_results[0], 4), round(author_results[1], 4))

#%%
test_size = 0.25
print(f'\nAnalyzing tweets at the author level, test_size={test_size}')
min_df=500
author_results = analyze_tweets(
    tweets_per_author, 'E', classifier=NAIVE_BAYES, min_df=min_df,
    get_words_and_probas=True, test_size=test_size)
print(f'Full set, min_df={min_df} training and test scores:', end=' ')
print(round(author_results[0], 4), round(author_results[1], 4))

