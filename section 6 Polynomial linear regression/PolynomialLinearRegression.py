#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 23:18:34 2018

@author: taojing
"""

# Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values

# no need to split train test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#no need for feature scaling

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
ploy_reg = PolynomialFeatures(degree = 4)
X_ploy = ploy_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_ploy, y)
