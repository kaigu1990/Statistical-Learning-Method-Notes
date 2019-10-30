# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:10:07 2019

@author: kgu
"""

import numpy as np
import pandas as pd
from collections import Counter

#X = np.array([[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],[2,2,1,1,2,2,2,1,2,2,2,2,1,1,2],[2,2,2,1,2,2,2,1,1,1,1,1,2,2,2],[3,2,2,3,3,3,2,2,1,1,1,2,2,1,3]])
#y = [2,2,1,1,2,2,2,1,1,1,1,1,1,1,2]

dataset = pd.read_table("tree.txt")
dataset = np.array(dataset)

X = dataset[:,:-1]
y = dataset[:,-1]
label_cls = np.unique(y)

def calc_sub_Gini(y_sub):
    gini_sub = []
    for cls in set(y_sub):
        gini_sub.append((np.sum(y_sub == cls) / len(y_sub)) ** 2)
    return 1- sum(gini_sub)

def cal_best_feature(dataset, label):
    feature_gini = {}
    for feature in range(dataset.shape[1]):
        for feature_arr in set(dataset[:,feature]):
            index1 = X[:,feature] == feature_arr
            index2 = X[:,feature] != feature_arr
            
            gini = sum(index1) / len(index1) * calc_sub_Gini(label[index1]) + sum(index2) / len(index2) * calc_sub_Gini(label[index2])
            feature_gini[(feature, feature_arr)] = gini
#            print(feature, feature_arr, gini)
    best_feature = min(feature_gini, key = feature_gini.get)
    return best_feature


def create_Tree(dataset, label, feature, limits):
    if (len(np.unique(label)) == 1):
        return label[0]
    
    
    if (len(dataset[0] == 0) || gini < limits):
        
