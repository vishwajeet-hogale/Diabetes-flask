# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:18:00 2020

@author: vishw
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import pickle 
warnings.filterwarnings("ignore")

# Importing the dataset
dataset = pd.read_csv('Diabetes.csv')

X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

pickle.dump(classifier,open('model.pkl','wb'))
