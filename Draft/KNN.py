# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:18:36 2019

@author: kgu
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold # import KFold

X = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
label = ["A","A","B","B"]
z = [0.1,0]
k = 3

dataset = pd.read_csv("train.csv")
dataset = np.array(dataset)
sample_num = dataset.shape[0]

train_dat = dataset[:,1:]
train_label = dataset[:,0]

#train_dat = dataset[:int(sample_num*0.9),1:]
#train_label = dataset[:int(sample_num*0.9),0]

test_dat = dataset[int(sample_num*0.9):,1:]
test_label = dataset[int(sample_num*0.9):,0]

class KNN:
    def brute_force_knn(self, dat_X, label, input_x, k):
        # 计算欧式距离
        dist = ((input_x - dat_X)**2).sum(axis=1)**0.5
        dist_index = np.argsort(dist)
        # 分类决策：多数表决
        label_pre = []
        for i in dist_index[:k]:
            label_pre.append(label[i])
        target = Counter(label_pre).most_common(1)[0][0]
        return target

    def knn_fd_cv(self, dat_X, label, k, fd):
        kf = KFold(n_splits=fd)
        error_list = []
        for train_index, test_index in kf.split(dat_X):
            X_train, X_test = dat_X[train_index], dat_X[test_index]
            y_train, y_test = label[train_index], label[test_index]
            
            error = 0
            for i in range(len(y_test)):
                t = knn.brute_force_knn(dat_X=X_train, label=y_train, input_x=X_test[i], k=k)
                # 计算测试误差
                if t != y_test[i]:
                    error += 1
            print(error/len(y_test)*100)
            error_list.append(error/len(y_test)*100)
        return np.mean(error_list)


knn = KNN()

res = []
for i in range(len(test_label)):
    t = knn.brute_force_knn(dat_X=train_dat, label=train_label, input_x=test_dat[i], k=5)
    res.append(1) if t == test_label[i] else res.append(0)
        
sum(res)/len(res)*100

kf = KFold(n_splits=10)
#kf.get_n_splits(dataset)
error_list = []
for train_index, test_index in kf.split(test_dat):
    X_train, X_test = test_dat[train_index], test_dat[test_index]
    y_train, y_test = test_label[train_index], test_label[test_index]
    
    error = 0
    for i in range(len(y_test)):
        t = knn.brute_force_knn(dat_X=X_train, label=y_train, input_x=X_test[i], k=5)
        # 计算测试误差
        if t != y_test[i]:
            error += 1
    print(error/len(y_test)*100)
    error_list.append(error/len(y_test)*100)

r = knn.knn_fd_cv(dat_X=test_dat, label=test_label, k=5, fd=10)

for i in range(2,10):
    r = knn.knn_fd_cv(dat_X=train_dat, label=train_label, k=i, fd=10)
    print("k =",i,"error rate =",r)


#from sklearn.model_selection import KFold # import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array
y = np.array([1, 2, 3, 4]) # Create another array
kf = KFold(n_splits=2) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf) 

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


#knn.brute_force_knn(dat_X=X, label=label, input_x=z, k=3)

#data_set_size = X.shape[0]
#diff_mat = np.tile(z, (data_set_size, 1)) - X
#sq_diff_mat = diff_mat**2
#sq_distances = sq_diff_mat.sum(axis=1)
#distances = sq_distances**0.5

dist = ((z - X)**2).sum(axis=1)**0.5
#dist_index = dist.argsort()
dist_index = np.argsort(-dist)
#Counter(label[dist_index[:k]])
label_pre = []
for i in dist_index[:k]:
    label_pre.append(label[i])
target = Counter(label_pre).most_common(1)[0][0]
print(target)