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
