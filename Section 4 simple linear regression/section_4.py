#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:14:47 2018

@author: taojing
"""

import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=0)

# Fitting simple linear regression model
from sklearning.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X.train, y_train)