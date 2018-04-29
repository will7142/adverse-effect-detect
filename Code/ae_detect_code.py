#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 05 09:30:14 2018

@author: will7142 (Will Rosenfeld)

Purpose: To identify the existance of adverse effects in doctors notes.
"""

# imports
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer

# read notes with AEs to DataFrame
ae = pd.read_csv('../data/DRUG-AE.txt',sep='|', header=None)
ae = ae.drop(labels=[0,2,3,4,5,6,7], axis = 1)
ae.drop_duplicates(inplace=True)
ae.columns = ['text']
ae['ae'] = 1

# read notes with no AEs to DataFrame
ne = pd.read_csv('../data/ADE-NEG.txt',sep='|', header=None)
ne.columns = ['pre']
ne[['pre','text']] = ne['pre'].str.split('NEG ',expand=True)
ne = ne.drop(labels=['pre'], axis = 1)
ne['ae'] = 0

# combine DataFrames
df = ae.append(ne, ignore_index=False, verify_integrity=False)
df.shape

# define X and y
X = df.text
y = df.ae

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# tokenization
vect = CountVectorizer(ngram_range=(1, 2))

# create document term matrix
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# use Naive Bayes to predict existance of adverse effect
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

# evaluate model
print(metrics.accuracy_score(y_test, y_pred_class)) #accuracy

# null accuracy
y_test_binary = np.where(y_test==1, 1, 0)
print(y_test_binary.mean())
print(1 - y_test_binary.mean())













