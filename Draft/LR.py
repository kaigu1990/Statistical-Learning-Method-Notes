# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:23:41 2019

@author: kgu
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_dataset = pd.read_table("horseColicTraining.txt", header = None)
test_dataset = pd.read_table("horseColicTest.txt", header = None)
train_dataset = np.mat(train_dataset)
test_dataset = np.mat(test_dataset)

train_X = train_dataset[:,:-1]
train_y = train_dataset[:,-1]
train_X = np.insert(train_X, train_X.shape[1], values = 1, axis = 1)

test_X = test_dataset[:,:-1]
test_y = test_dataset[:,-1]
test_X = np.insert(test_X, test_X.shape[1], values = 1, axis = 1)

dataset = pd.read_table("testSet.txt", header = None)
dataset = np.mat(dataset)
X = dataset[:,:-1]
y = dataset[:,-1]
X = np.insert(X, X.shape[1], values = 1, axis = 1)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 123456)

class logisticRegression:
    # Sigmoid函数
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    # 在对数似然函数导数前面加了负号，因此用梯度下降法，运用矩阵计算，迭代1000次，求解w
    def gradient_descent(self, X, y, item = 1000, alpha = 0.001):
        w = np.ones((X.shape[1],1))
        for i in range(item):
            h = self.sigmoid(np.dot(X, w))
            error = h - y
            w -= alpha * np.dot(X.T, error) 
        return w
    
    # 计算wx，通过公式6.3确定分类值
    def predict(self, w, test):
        p1 = self.sigmoid(np.dot(test, w))
        if p1 >= 0.5:
            return 1
        else:
            return 0

alpha = 0.001
item = 1000

LR = logisticRegression()

w = LR.gradient_descent(train_X, train_y)

error = 0
for i in range(len(test_X)):
    predict_y = LR.predict(w, test_X[i,:])
    if predict_y != test_y[i]:
        error += 1
error / len(test_y) * 100



#def sigmoid(z):
#    return 1/(1+np.exp(-z))
#
#def gradient_descent(X, y, item = 1000, alpha = 0.001):
#    w = np.ones((X.shape[1],1))
#
#    for i in range(item):
#        h = sigmoid(np.dot(X, w))
#        error = h - y
#        w -= alpha * np.dot(X.T, error) 
#    return w
#
#def predict(w, test_X):
#    p1 = sigmoid(np.dot(test_X,w))
#    if p1 >= 0.5:
#        return 1
#    else:
#        return 0

alpha = 0.001
item = 1000

LR = logisticRegression()

w = LR.gradient_descent(train_X, train_y)

error = 0
for i in range(len(test_X)):
    predict_y = LR.predict(w, test_X[i,:])
    if predict_y != test_y[i]:
        error += 1
error / len(test_y) * 100
    
