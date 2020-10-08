# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 08:34:34 2020

@author: Vikee
"""

#  SMS Spam Collection Data Set - Spam Classifier


import pandas as pd

messages = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])

#Data cleaning and preprocessing

import re
import nltk


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i]) # except  characters everything replaces with space
    review = review.lower()
    review = review.split() # split sentence to get list of words
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # stemming
    review = ' '.join(review)
    corpus.append(review)


# Creating the Bag of Words model
    
    # without (max_features=2500) , we get 6296 cols in X
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500) # we limit cols to 2500. more frequently present words
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values # take one col


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)


from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test,y_pred)




















