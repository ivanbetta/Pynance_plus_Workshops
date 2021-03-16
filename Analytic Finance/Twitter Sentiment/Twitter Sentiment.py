
# Twitter Sentiment

"""
The objective of this code is to get twitters sentiment from stocks.
"""

# Libraries

import tweepy as tp
from textblob import *
import jsonpickle
import pandas as pd
import time
import json
import os
import re
from textblob import TextBlob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

# Functions

def clean(tweet):
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'@[A-Za-z0–9]+', '', tweet)
    tweet = re.sub(r'[^A-Za-z0-9]+', ' ', tweet)
    tweet = re.sub(r'[0-9]', '', tweet)
    tweet = tweet.lower()
    list_of_words = ['b ',' status ',' api ','api ',' tweepy ',' object ',' id ',' str ',
                     ' at ',' json ',' created ',' mon ',' tue ',' wed ',' thu ',
                     ' fri ',' sat ',' sun ',' jun ',' feb ',' mar ',' apr ',' may ',
                     ' jun ',' jul ',' aug ',' sep ',' oct ',' nov ',' dic ',' text ',
                     ' xf ', ' x ', ' xja ', ' n ', ' xe ', ' xa ']
    for word in list_of_words:
        tweet = re.sub(word, ' ', tweet)    
    tweet = re.sub(' +', ' ', tweet)    
    tweet_words = tweet.split()
    try:
        tweet_words.pop(0)  
    except:
        pass
    tweet = ' '.join(tweet_words)
    if tweet == '' or tweet == ' ':
        pass
    else:
        return tweet

def read_tweets(file_name):
    with open(file_name,'r') as f:
        tweets = [clean(line.strip()) for line in f]
    f.close()
    return tweets

# Tokens

TOKEN = "AAAAAAAAAAAAAAAAAAAAAKD8NAEAAAAAIL6rH%2FtfYMuMXG9bhDuzvlcyeFM%3DyzJq5kdYipJVvwaGuV0we6DIdDA8PR0gSVyryvImxSAebDWOPI"
SECKEY = "G0u6Qi3LF7w8conui9fznxXms0uqD9DlkKoDSHZzQdPR7Fs9G7"
APIK = "p3CTl7fWCaSuZ5p8DT3OYPDeV"

ACCES_TOKEN = "1365316010505621509-BnR829L59HLcDZx3rGPGyZW6Uv9odn"
ACCES_SECRET = "DPJd6NjLIkv2ZXbJFdX6lQ5i6m2bI7vTnm9mWzJBOWcoP"

auth = tp.OAuthHandler(APIK,SECKEY)
auth.set_access_token(ACCES_TOKEN,ACCES_SECRET)
auth.secure = True
api = tp.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

#Whatever we want to search about
searchQuery = "$BABA"
retweet_filer = " -filter:retweets"
#With this you can also filter user accounts, places...
que = searchQuery + retweet_filer
tweetsPerQry = 100
fName = "tweets.txt"

# Main Program

FECHAHOY=date.today()
FECHAHOY=str(FECHAHOY)
idiom = "en"
users_locs=[]
user_loc=[]
coord=[]
with open(fName,'w') as f:
#opening a writing text fot tweets
    tweets = []
    try:
        new_tweets = tp.Cursor(api.search,
               q=que,
               lang="en",
               since=FECHAHOY).items(1000)
        #Query for searching
        if not new_tweets:
            print("No more tweets found")

        for tweets in new_tweets:
  
            # if (tweets.coordinates is not None): 
            #     lon = tweets.coordinates['coordinates'][0]
            #     lat = tweets.coordinates['coordinates'][1]
            #     coord.append([lon,lat])
            users_loc = [tweets.user.screen_name, tweets.user.location,tweets.coordinates]
            #Obtaining location
            users_locs.append(users_loc)
            tweet=tweets.text
            tweet = str(tweet)
            tweet = re.sub(r'^RT[\s]+', ' ', tweet)
            tweet = re.sub(r'https?:\/\/.*[\r\n]*', ' ', tweet)
            tweet = tweet.lower()
            tweet = re.sub(r'[^a-z-ñáéíóú]+', ' ', tweet)


            if 'text' in tweet:
                tweet_words = tweet.split()
                for text_word in range(len(tweet_words)):
                    if tweet_words[text_word] == "text":
                        tweet = ' '.join(tweet_words[text_word+1:])
            print(tweet)
            f.write( str(tweet.replace('\n',''))+"\n")

    except tp.TweepError as e:
        print("Errr:" + str(e))

tweets = read_tweets(fName)
# print(tweets[2])
# print(TextBlob(tweets[2]).sentiment)

polarity = lambda x: TextBlob(x).sentiment.polarity
subjectivity = lambda x: TextBlob(x).sentiment.subjectivity
#Using textblob to get polarity and subjectivity
tweet_polarity = np.zeros(len(tweets))
tweet_subjectivity = np.zeros(len(tweets))
contador=0
for idx, tweet in enumerate(tweets):

    tt=str(tweet)
    tweet_polarity[idx] = polarity(tt)
    tweet_subjectivity[idx] = subjectivity(tt)

sns.scatterplot(tweet_polarity, # X-axis
                tweet_subjectivity,  # Y-axis
                s=100);
#Getting the sentiment graphs
plt.title("Sentiment Analysis "+ str(searchQuery), fontsize = 20)
plt.xlabel('← Negative — — — — — — Positive →', fontsize=15)
plt.ylabel('← Facts — — — — — — — Opinions →', fontsize=15)
plt.tight_layout()

f, axs = plt.subplots(1, 2, figsize=(15,5))

sns.distplot(tweet_polarity, color="b", ax=axs[0])
axs[0].set_title("Tweet Polarity", fontsize = 20)
axs[0].set_xlabel('← Negative — — — — Positive →', fontsize=15)
sns.distplot(tweet_subjectivity, color="b", ax=axs[1])
axs[1].set_title("Tweet Subjectivity", fontsize = 20)
axs[1].set_xlabel('← Facts — — — — — Opinions →', fontsize=15)

plt.tight_layout()

# References

'https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/twitter-data-in-python/'
'https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query'