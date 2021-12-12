# # -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:35:28 2021

Step 1: Create a CSV of tweets from the list of top authors per MBTI type. The
script screens out apparent retweets and non-English speakers. From the
remaining tweets, the script stores 102 tweets (a number divisible by 3) from
each of the 100 users per Myers-Briggs type, for a total of 163,200 tweets.
These tweets are exported to CSV files, one per type, for later analysis.

@author: Barrett
"""

import re
import sys
LIBPATH = r'D:\Springboard_DataSci\Assignments\Lib'
if LIBPATH not in sys.path:
    sys.path.insert(0, LIBPATH)
import pandas as pd
from langdetect import detect, LangDetectException
import tweepy
import Tweepy_keys
import TimeTracker

USERS_PER_MBT = 100 
TWEETS_PER_USER = 102 #Ideally, divisible by 3
DF_COLUMNS = ['Screen name', 'MBTI', 'Time', 'Tweet']

# Get all the keys, tokens, and authorizations.
consumer_key = Tweepy_keys.consumer_key
consumer_secret = Tweepy_keys.consumer_secret
access_token = Tweepy_keys.access_token
access_token_secret = Tweepy_keys.access_token_secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Get the API.
api = tweepy.API(auth, wait_on_rate_limit=True)
path = r'D:\Springboard_DataSci\MBTI_project'

# Set up the 16 types.
letters = [['E', 'I'], ['S', 'N'], ['F', 'T'], ['J', 'P']]
MB_types = []
for i in range(16):
    MB_types.append(letters[0][i//8%2] + letters[1][i//4%2]
                     + letters[2][i//2%2] + letters[3][i%2])

stopwatch = TimeTracker.TimeTracker() # Don't include setup.
for i, MB_type in enumerate(MB_types):
    # Get all the bios and users of the current MB type.
    DF_of_tweets_exists = False
    bios = pd.read_csv(path + '\\Data Input\\' + MB_type + '_bios.csv')
    print('\nCollecting', MB_type, 'tweets:', i+1, 'out of', len(MB_types))
    users = 0
    df_tweets = pd.DataFrame(columns=DF_COLUMNS)
    
    for j, handle in enumerate(bios['Screen name']):
        tweets = []
        # Get the next user of that type.
        bio = str(bios['Bio'].iloc[j]) # str() avoids issues with nan
        non_current_types = list(set(MB_types).difference({MB_type}))
        if any(other_type in bio for other_type in non_current_types):
            print(handle + ' may identify as multiple Myers-Briggs types, not '
                  + 'just ' + MB_type + '. Skipping this user.')
            continue

        count_eng = count_non_eng = 0
        try:
            n_useful_tweets = 0
            for status in tweepy.Cursor(
                    api.user_timeline, screen_name='@' + handle,
                    tweet_mode="extended").items():
                tweet = status.full_text
                # Filter out any of the following:

                # 1. Retweets and tweets with fewer than five words
                if 'http' in tweet: #Trim out any web addresses.
                    tweet = tweet[
                        :[m.start() for m in re.finditer('http', tweet)][0]]
                if tweet[:2] == 'RT' or len(tuple(tweet.split(' '))) < 5:
                    continue

                # 2. Tweets with an undecypherable langauge
                try:
                    lang = detect(tweet)                
                except LangDetectException:
                    continue
                
                # Denote whether the tweet appears to be in English.
                if detect(tweet) == 'en':
                    count_eng += 1
                    n_useful_tweets += 1
                    tweets.append([handle, MB_type, status.created_at,
                                   status.full_text])
                else:
                    count_non_eng += 1
                
                # If most of the tweets are in English, save them.
                if (count_eng+count_non_eng) % (USERS_PER_MBT//3) == 0:
                    likelyEnglishUser = count_eng/(count_eng+count_non_eng)\
                        > 0.75
                    if not likelyEnglishUser:
                        print('\t', handle, 'may be a non-English speaker. '
                              'Skipping to the next one.')
                        break
                if n_useful_tweets >= TWEETS_PER_USER:
                    print("\tChecking user's language...passed."
                          " Saving tweets of", handle)
                    break
            else:
                print('No more tweets for this user. Skipping to next')
                continue
    
        except tweepy.TweepError:
            print("\tDownloading " + handle + "'s tweets failed.",
                  'Proceeding to next one')
        else:
            if likelyEnglishUser:
                users += 1
                print(users, 'out of', USERS_PER_MBT, 'complete for', MB_type)
                df_tweets = df_tweets.append(
                    pd.DataFrame(tweets, columns=DF_COLUMNS),
                    ignore_index=True)
                if users >= USERS_PER_MBT:
                    break
            
    print('Elapsed time for ' + MB_type + ': ' + stopwatch.getElapsedTime())
    df_tweets.to_csv(path + '\\Data Output\\' + MB_type + '_tweets.csv',
                      index=False)