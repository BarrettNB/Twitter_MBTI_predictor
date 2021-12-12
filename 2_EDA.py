# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 18:46:23 2021

@author: Barrett
"""

#%% Setup
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from scipy import sparse
import numpy as np
import pandas as pd
import os

TYPE = 'Type'
LOGISTIC = 'logistic'
NAIVE_BAYES = 'multinomial naive Bayes'

# Get all the tweets.
path = r'D:\Springboard_DataSci\MBTI_project\Data Output'
os.chdir(path)

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
   
#%% Analyze tweets; sort words by score
def analyze_tweets(tweets, letter, classifier, min_df=200, max_df=1.,
                   alpha=1., C=1, stop_words=None, get_words_and_probas=False):
    '''Text classification of the tweets'''
    y = tweets[letter]
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df,
                                 stop_words=stop_words)
    tweets = tweets['Tweet'].to_list()
    # Get the sparse matrix (x, y) of (tweetID, wordID).
    X = vectorizer.fit_transform(tweets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    if classifier==LOGISTIC:
        clf = LogisticRegression(C=C, max_iter=1e3, random_state=0)\
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
        
def sort_words_by_coef(words, coefs):
    words_order = np.argsort(coefs)
    return words[words_order], coefs[words_order]

#%% Logistic regression
print('\nAnalyzing tweets with logistic regression')
for min_df in [10, 25, 50, 100, 250, 500]:
    score_base = analyze_tweets(tweets, 'E', classifier=LOGISTIC,
                                min_df=min_df)
    print(f'Full set, min_df={min_df} training and test scores:', end=' ')
    print(round(score_base[0], 4), round(score_base[1], 4))
    
#%% Naive Bayes
'''Lower min_dfs give better fits but also cause overfitting. Let's retry with
multinomial naive Bayes and see what kinds of results we get.'''
print('\nAnalyzing tweets with naive Bayes')
for min_df in [10, 25, 50, 100, 250, 500]:
    score_base = analyze_tweets(tweets, 'E', classifier=NAIVE_BAYES,
                                min_df=min_df)
    print(f'Full set, min_df={min_df} training and test scores:', end=' ')
    print(round(score_base[0], 4), round(score_base[1], 4))

#%% Naive Bayes, no common words
'''This is much better. Let's get rid of common words and rerun with
min_df=10.'''
print('\nAnalyzing tweets with naive Bayes and stop words removed')
for min_df in [10, 25, 50, 100, 250, 500]:
    score_base = analyze_tweets(tweets, 'E', classifier=NAIVE_BAYES,
                                min_df=min_df, stop_words=ENGLISH_STOP_WORDS)
    print(f'No stop words, min_df={min_df} training and test scores:', end=' ')
    print(round(score_base[0], 4), round(score_base[1], 4))

#%% Naive Bayes, no common words, get features
'''Scores are marginally hurt, but we have fewer features to worry about.
This is an acceptable trade. Let's see what the features are.'''
min_df=10
print(f'\nNaive Bayes, stop words removed, min_df={min_df}')
bayes_results = analyze_tweets(
    tweets, 'E', classifier=NAIVE_BAYES, min_df=min_df,
    stop_words=ENGLISH_STOP_WORDS, get_words_and_probas=True)
print('Scores:', round(bayes_results[0], 4), round(bayes_results[1], 4))
nb_words = bayes_results[2]
nb_coefs = bayes_results[3]
nb_words, nb_coefs = sort_words_by_coef(nb_words, nb_coefs)

'''A lot of these are nonsense, probably other users. We need to rerun this
with a much higher min_df, even at the cost of scores. Let's also lower the
value of alpha in the naive Bayes to get stronger regularization.'''
min_df=200; alpha=5
print(f'\nNaive Bayes, stop words removed, min_df={min_df}, alpha={alpha}')
bayes_results = analyze_tweets(
    tweets, 'E', classifier=NAIVE_BAYES, min_df=min_df, alpha=alpha,
    stop_words=ENGLISH_STOP_WORDS, get_words_and_probas=True)
print('Scores:', round(bayes_results[0], 4), round(bayes_results[1], 4))
nb_words = bayes_results[2]
nb_coefs = bayes_results[3]
nb_words, nb_coefs = sort_words_by_coef(nb_words, nb_coefs)

'''Perhaps returning the stop words and lowering min_df will help.'''
min_df=100; alpha=5
print(f'\nNaive Bayes, stop words returned, min_df={min_df}, alpha={alpha}')
bayes_results = analyze_tweets(
    tweets, 'E', classifier=NAIVE_BAYES, min_df=min_df, alpha=alpha,
    get_words_and_probas=True)
print('Scores:', round(bayes_results[0], 4), round(bayes_results[1], 4))
nb_words = bayes_results[2]
nb_coefs = bayes_results[3]
nb_words, nb_coefs = sort_words_by_coef(nb_words, nb_coefs)

#%% Grouping by author instead of by tweet
'''These results leave a lot to be desired. One possibility is that when one
author tweets a lot of words, they unduly weight particular words in their
favor. Let's combine tweets per author.'''

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

#%% Naive Bayes with combined tweets
'''Let's rerun with combined tweets. We try a lower min_df because of the
combined words.'''
min_df=50
print(f'\nNaive Bayes, min_df={min_df}')
bayes_results = analyze_tweets(
    tweets_per_author, 'E', classifier=NAIVE_BAYES, min_df=min_df,
    get_words_and_probas=True)
print('Scores:', round(bayes_results[0], 4), round(bayes_results[1], 4))
nb_words = bayes_results[2]
nb_coefs = bayes_results[3]
nb_words, nb_coefs = sort_words_by_coef(nb_words, nb_coefs)

'''Badly overfit. Let's change min_df.'''
min_df=200
print(f'\nNaive Bayes, stop words removed, min_df={min_df}')
bayes_results = analyze_tweets(
    tweets_per_author, 'E', classifier=NAIVE_BAYES, min_df=min_df,
    get_words_and_probas=True)
print('Scores:', round(bayes_results[0], 4), round(bayes_results[1], 4))
nb_words = bayes_results[2]
nb_coefs = bayes_results[3]
nb_words, nb_coefs = sort_words_by_coef(nb_words, nb_coefs)

'''Less bad but still problematic. Let's experiment with alpha.'''
min_df=200; alpha=10
print(f'\nNaive Bayes, min_df={min_df}, alpha={alpha}')
bayes_results = analyze_tweets(
    tweets_per_author, 'E', classifier=NAIVE_BAYES, min_df=min_df, alpha=alpha,
    get_words_and_probas=True)
print('Scores:', round(bayes_results[0], 4), round(bayes_results[1], 4))
nb_words = bayes_results[2]
nb_coefs = bayes_results[3]
nb_words, nb_coefs = sort_words_by_coef(nb_words, nb_coefs)

min_df=200; alpha=0.1
print(f'\nNaive Bayes, min_df={min_df}, alpha={alpha}')
bayes_results = analyze_tweets(
    tweets_per_author, 'E', classifier=NAIVE_BAYES, min_df=min_df, alpha=alpha,
    get_words_and_probas=True)
print('Scores:', round(bayes_results[0], 4), round(bayes_results[1], 4))
nb_words = bayes_results[2]
nb_coefs = bayes_results[3]
nb_words, nb_coefs = sort_words_by_coef(nb_words, nb_coefs)

'''Perhaps min_df is the issue?'''
min_df=500
print(f'\nNaive Bayes, min_df={min_df}')
bayes_results = analyze_tweets(
    tweets_per_author, 'E', classifier=NAIVE_BAYES, min_df=min_df,
    get_words_and_probas=True)
print('Scores:', round(bayes_results[0], 4), round(bayes_results[1], 4))
nb_words = bayes_results[2]
nb_coefs = bayes_results[3]
nb_words, nb_coefs = sort_words_by_coef(nb_words, nb_coefs)

#%% Analyzing the split
def get_word_use(words, tweets_with_words, trait, other_trait):
    '''Split totals of unique authors who tweeted these words.
    Trait needs to be E/S/F/J; other_trait, I/N/T/P.'''
    unique_words = pd.DataFrame(columns=['Word', trait, other_trait])
    for i, word in enumerate(words):
        # Column 1: Whether each author tweeted the word of interest.
        word_in_tweets = tweets_per_author['Tweet'].str.contains(word)\
            .to_frame()
        # Column 2: Whether that author 
        word_in_tweets[trait] = tweets_per_author[trait].to_numpy()
        # Now get the summaries. Set the index to the trait boolean
        word_use = word_in_tweets.value_counts().reset_index()\
            .set_index(trait)
        # Only keep the ones with the word actually in them
        word_use = word_use[word_use['Tweet']]
        # Add this info to unique_words
        unique_words.loc[i] = [word, word_use.loc[True, 0],
                               word_use.loc[False, 0]]
    # Column 1 is the trait; column 2 is the "not trait."
    unique_words['Percent ' + trait] = 100*unique_words[trait] /\
        (unique_words[trait]+unique_words[other_trait])
    return unique_words

#%% Get unique words
unique_words_EI = get_word_use(nb_words, tweets_per_author, trait='E',
                               other_trait='I')

#%% Unique words for other letters
min_df=250
print(f'\nNaive Bayes, min_df={min_df}')
bayes_results = analyze_tweets(
    tweets_per_author, 'S', classifier=NAIVE_BAYES, min_df=min_df,
    get_words_and_probas=True)
print('Scores:', round(bayes_results[0], 4), round(bayes_results[1], 4))
nb_words = bayes_results[2]
nb_coefs = bayes_results[3]
nb_words, nb_coefs = sort_words_by_coef(nb_words, nb_coefs)

print('Calculating unique words')
unique_words_SN = get_word_use(nb_words, tweets_per_author, trait='S',
                               other_trait='N')

'''Some percentages farther away from 50 start to show up. Perhaps we should
lower min_df after all?'''
min_df=150
print(f'\nNaive Bayes, min_df={min_df}')
bayes_results = analyze_tweets(
    tweets_per_author, 'S', classifier=NAIVE_BAYES, min_df=min_df,
    get_words_and_probas=True)
print('Scores:', round(bayes_results[0], 4), round(bayes_results[1], 4))
nb_words = bayes_results[2]
nb_coefs = bayes_results[3]
nb_words, nb_coefs = sort_words_by_coef(nb_words, nb_coefs)

print('Calculating unique words')
unique_words_SN = get_word_use(nb_words, tweets_per_author, trait='S',
                               other_trait='N')

'''With a lower min_df, more extreme percentages show up, but the number of
authors per word drops. It seems we have a trade-off to make.'''