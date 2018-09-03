#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:14:47 2018

@author: taojing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

# X=dataset.iloc[:,0].values is wrong , because it gives a vector not a matrix with 1 comlum
# It matters when fitting the regession model

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=0)


X_train_reshape = X_train.reshape(-1,1)

# Fitting simple linear regression model
from sklearn.linear_model import LinearRegression

#X_train.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
#visualize

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()
