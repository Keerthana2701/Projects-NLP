# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:28:18 2020

@author: Vikee
"""


import tweepy
import re
import pickle

from tweepy import OAuthHandler  # authenticating our machine with twitter server
total_pos = 0
total_neg = 0
# Initializing the keys

consumer_key = 'ypVim69JzGoePTbyZk9pOr55B'
consumer_secret = '1JVx8bHohG3YbeIR9HHQqnYXuwV3y4Zs9TOJNBBQEvU9j0qlcT'
access_token = '1273394433136836610-bzO0tjUmGOiO7oYZAGr43GFkyGEgzn'
access_secret ='43ZbafDcYK58FTdQ5aTSmFkTH8iCVHPtPLSFHs5IK2Q2G'

# Initializing the tokens

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
args = ['trump'];
api = tweepy.API(auth,timeout=10)

# Fetching the tweets

list_tweets = []
query = args[0]

if len(args) == 1:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='recent',geocode="22.1568,89.4332,500km").items(100):
        list_tweets.append(status.text)
        
        
# Loading the vectorizer and classfier
with open('classifier.pickle','rb') as f:
    classifier = pickle.load(f)
    
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)  
    
    
  # Preprocessing the tweets
for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    print(tweet)   
    
    sent = classifier.predict(tfidf.transform([tweet]).toarray())
    print(tweet,":",sent)
    if sent[0] == 1:
        total_pos += 1
    else:
        total_neg += 1
        
        
 