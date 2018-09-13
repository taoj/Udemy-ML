#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 22:19:10 2018

@author: taojing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# encode categorical column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X= LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

#Avoiding dummpy variable trap
X = X[: , 1:]

# split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling , algorithum did it for us

#fit multiple variable linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting test_set result
y_pred = regressor.predict(X_test)
plt.scatter(y_test, y_pred)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
# pay attention capitalization !
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# x2 has highest P value 0.99 , delete it and run it again
X_opt = X[:,[0,1,3,4,5]]
# pay attention capitalization !
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# x1 has highest P value 0.94 , delete it and run it again
X_opt = X[:,[0,3,4,5]]
# pay attention capitalization !
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# x2 has highest P value 0.602 , delete it and run it again
X_opt = X[:,[0,3,5]]
# pay attention capitalization !
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# x2 has highest P value 0.060 , delete it and run it again
X_opt = X[:,[0,3]]
# pay attention capitalization !
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()



