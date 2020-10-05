# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:15:14 2020

@author: Vikee
"""


import numpy as np
import re
import pickle 
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

reviews = load_files('txt_sentoken/') #-loops through sub folders. for neg generates class 0 for pos it generates class 1
X,y = reviews.data,reviews.target

# Pickling the dataset

with open('X.pickle','wb') as f: #write byte
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)

# Unpickling dataset
    
with open('X.pickle','rb') as f: #read byte
    X=pickle.load(f)
    
with open('y.pickle','rb') as f:
    y=pickle.load(f)
    

# Creating the corpus
    
corpus = []
for i in range(0, 2000):
    review = re.sub(r'\W', ' ', str(X[i]))  # non word characters substituted with space
    review = review.lower()                 # convert to lower case
    review = re.sub(r'^[a-z]\s+',' ',review) # remove single characters preceded by space
    review = re.sub(r'\s+[a-z]\s+', ' ',review) # remove single characters followed or preceded by a space
    review = re.sub(r'\s+', ' ', review)   # remove single characters followed by space
    corpus.append(review)  
    
    
    
# Creating the BOW model
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english')) # max_df=0.6 words that appear in 60 percent of document, 
X = vectorizer.fit_transform(corpus).toarray()


# transform BOW model to tfidf using tfidftransformer

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()


# Creating the Tf-Idf model directly   
 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))  
X = vectorizer.fit_transform(corpus).toarray()



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



# Training the classifier using Logistic Regression  ( for binary classification)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

# here, each sentence is mapped to a point. if the point is > 0.5, then positive else negative
# logistic regression calculates the values of coefficients
# based on the coefficients, new sentences are given points
#
# here x1 to x2000 i.e 2000 independent features and y is dependent feature
# y= a_bx1+cx2+......dx2000 , a,b,c,d are coefficients


# Testing model performance

sent_pred = classifier.predict(text_test)

# testing accuracy

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test, sent_pred)




# Saving our classifier

with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
# Saving the Tf-Idf model   - classifier.predict needs vectorized input. so save the vectorizer
# pickle the vectorizer
    
    
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
    
    # hence we get classifier.pickle and tfidfmodel.pickel
    # we can use this as pretrained models for other analysis like sentiment analysis



# Using our classifier  - import the above pickle files
    
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
    
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)           
    
    
sample = ["You are a nice person man, have a good life"]
sample = tfidf.transform(sample).toarray()  # no need to fit. just transform using that pickle pretrained model

sentiment = clf.predict(sample)


sample1 = ["You are a bad person man, have a worst life"]
sample1 = tfidf.transform(sample1).toarray()  # no need to fit. just transform using that pickle pretrained model

sentiment1 = clf.predict(sample1)







