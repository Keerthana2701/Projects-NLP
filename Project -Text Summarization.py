# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:03:05 2020

@author: Vikee
"""


import bs4 as bs
import urllib.request
import re
import nltk
import heapq

# Gettings the data source

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming').read()

# we get complete html doc of this web page

# now we need string data as we cannot use html data

# so use beautiful soup and lxml libraries

# lxml is  a parser that beautiful source uses


# Parsing the data/ creating BeautifulSoup object

soup = bs.BeautifulSoup(source,'lxml') # gives clear html

# Fetching the text data 

text = ""
for paragraph in soup.find_all('p'): # p is html paragraph tag. text is always inside the paraagraph tag in wikipedia
    text += paragraph.text
    
    # this artilcle containes more impurities so preprocess it

# Preprocessing the data
    
text = re.sub(r'\[[0-9]*\]',' ',text) # remove numbers since they are references in wiki page
text = re.sub(r'\s+',' ',text)  
clean_text = text.lower()
clean_text = re.sub(r'\W',' ',clean_text)  # remove non word characters
clean_text = re.sub(r'\d',' ',clean_text)   # remove digits
clean_text = re.sub(r'\s+',' ',clean_text)   # remove extra spaces


# Tokenize sentences

sentences = nltk.sent_tokenize(text)  # use text and not clean text since it does not have anything except words

# Stopword list
stop_words = nltk.corpus.stopwords.words('english')






# Word counts  -- building histogram

word2count = {} # this dictionary will contin histogram
for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:  
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1 # is already there, just increase its count

# Converting counts to weights --- create weighted histogram - divide each by max value
            

for key in word2count.keys():
    word2count[key] = word2count[key]/max(word2count.values())

# this is the weighted histogram of each word
    # we get 1 if that word appears more no of times
    
    
    
    
    
 # Product sentence scores - to get the sentence score, we need to add word2ount score of all words in taht sentence
    
sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) < 30:  # sentences whose length of words i.e word count <30
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]
                    
                    
                    
 # Gettings best 5 sentences with high score using heapq library 
                    
best_sentences = heapq.nlargest(5, sent2score, key=sent2score.get)

print('---------------------------------------------------------')
for sentence in best_sentences:
    print(sentence)                   
    
    
    