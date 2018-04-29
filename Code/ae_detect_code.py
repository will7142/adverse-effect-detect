#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 09:30:14 2018

@author: will7142 (Will Rosenfeld)
"""

#IMPORTS
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
ae.shape
ne.shape
df = ae.append(ne, ignore_index=False, verify_integrity=False)
df.shape

# define X and y
X = df.text
y = df.ae

# split the new DataFrame into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)