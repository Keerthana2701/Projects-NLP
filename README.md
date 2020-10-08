# Projects--NLP



# 1. Text Classification

Text classification is an appliation of NLP where we create a model which classiifes human language into different classes

ex : gmail uses spam filter to classify text . which to put in spam folder based on data in the message

Also used in sentiment analysis

Cornell sentiment analysis dataset.- movie review data

http://www.cs.cornell.edu/people/pabo/movie-review-data/

polarity_dataset_v2.0 dataset

This dataset contains 1000 positive and 1000 negative reviews

Save the classifier and tfidf model as pickle file so that it can be used as pretrained model for twitter sentiment analysis project

# 2. Twitter Sentiment Analysis

use tweepy library

Create a twitter app and get the key values.

Initialize the keys and tokens 

OAuthHandler from tweepy is used for auternticating our machine with the twitter server

Fetch the tweets based on a particular word and preprocess the tweets


load the classifier and tfidf pickle files and use for predicting the preprocessed tweets

We get the positive anf negative words count of this tweet.


# 3. Text Summarization

tokenize paragraph from wiki page since the wiki page has text in form of paragraphs

wiki page to summarize : https://en.wikipedia.org/wiki/Global_warming

use urlib to  get complete html doc of this web page.

now we need string data as we cannot use html data

so use beautiful soup and lxml libraries

lxml is  a parser that beautiful source uses

parse the data using beautiful soup to get a better clean html data

from this fetch the text data

preprocess it.

tokenize the sentences

build histogram by taking the count for each words in all sentences

build weighted histogram by dividing  each count  by max value count

now we get weighted score for each word

find sentence score by adding the values of words in the sentence

now each sentence has a score

sort the sentence with high scores and take first n high sentence with max score


those sentences gives the summary of that wiki page



# 4. Spam Classifier with naive bayes model

dataset : https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

  preprocess the dats using re 
  
  bag of words model  and naive bayes MultinomialNB to predict the message as spam or ham
  
  print the accuracy and confusion matrix to find the correct predeitions
