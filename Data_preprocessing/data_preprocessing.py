#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:06:26 2018

@author: taojing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer()

imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])

np.set_printoptions(threshold = np.nan)

#encode categorical value
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

encoder_X = LabelEncoder()
X[:,0] = encoder_X.fit_transform(X[:,0])

oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()
encoder_y = LabelEncoder()
y = encoder_y.fit_transform(y);

#split dateset into training and testing 
from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = train_test_split(X,y,test_size=0,random_state = 0)
# scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train);
X_test = sc_X.transform(X_test);
# should scale dummy variable
# if scale it, everything will be on same scale , but lose interpretation
# if not scale, it won't break it







