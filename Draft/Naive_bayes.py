# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:03:38 2019

@author: kgu
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

X = np.array([[1,"S"],[1,"M"],[1,"M"],[1,"S"],[1,"S"],[2,"S"],[2,"M"],[2,"M"],[2,"L"],[2,"L"],[3,"L"],[3,"M"],[3,"M"],[3,"L"],[3,"L"]])
label = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]

label_value = np.unique(label)
p_label = {}
p_feature = {}
for lv in label_value:
    target = np.equal(lv, label)
    p_label[lv] = sum(target) / len(label)
#    print(lv, p_label)
    p_feature[lv] = {}
    for i in range(X.shape[1]):
        p_feature[lv][i] = {}
        for fl in np.unique(X[target][:,i]):
            p_feature[lv][i][fl] = np.sum(X[target][:,i] == fl) / len(label)
        
#        feature = X[target][i]
#        p_feature[tuple(feature)] = np.equal()


pri_p_label = {}
cond_p_feature = {}
for lv in set(label):
    t = np.equal(lv, label)
    pri_p_label[lv] = sum(t) / len(label)
    
    for i in range(X.shape[1]):
        for j in set(X[:,i]):
            cond_p_feature[(j,lv)] = np.sum(X[t][:,i] == j) / np.sum(t)

# =============================================================================
# 
# =============================================================================

# 拉普拉斯平滑
alpha = 0

# 整理分类
feature_data = defaultdict(lambda: [])
label_data = defaultdict(lambda: 0)
for feature, lab in zip(X, label):
    feature_data[lab].append(feature)
    label_data[lab] += 1

# 计算先验概率
pri_p_label = {k: (v + alpha)/(len(label) + len(np.unique(label)) * alpha) for k,v in label_data.items()}

# 计算不同特征值的条件概率
cond_p_feature = defaultdict(lambda: {})
for i,sub in feature_data.items():
    sub = np.array(sub)
    for f_dim in range(sub.shape[1]):
        for feature in np.unique(sub[:,f_dim]):
            cond_p_feature[i][(f_dim,feature)] = (np.sum(sub[:,f_dim] == feature) + alpha) / (sub.shape[0] + len(np.unique(sub[:,f_dim])) * alpha)
            
z = np.array([2,"S"])
p_data = {}
for sub_lable in np.unique(label):
    # 对概率值取log，防止乘积时浮点下溢
    p_data[sub_lable] = np.log(pri_p_label[sub_lable])
    for i in range(len(z)):
        p_data[sub_lable] *= np.log(cond_p_feature[sub_lable][(i,z[i])])
opt_label = max(p_data, key = p_data.get)

# =============================================================================
# 
# =============================================================================

class MultinomialNB:
    '''
    fit函数输入参数：
        X 测试数据集
        y 标记数据
        alpha 贝叶斯估计的正数λ
    predict函数输入参数：
        test 测试数据集
    '''
    def fit(self, X, y, alpha = 0):
        # 整理分类
        feature_data = defaultdict(lambda: [])
        label_data = defaultdict(lambda: 0)
        for feature, lab in zip(X, y):
            feature_data[lab].append(feature)
            label_data[lab] += 1

        # 计算先验概率
        self.label = y
        self.pri_p_label = {k: (v + alpha)/(len(self.label) + len(np.unique(self.label)) * alpha) for k,v in label_data.items()}
        
        # 计算不同特征值的条件概率
        self.cond_p_feature = defaultdict(lambda: {})
        for i,sub in feature_data.items():
            sub = np.array(sub)
            for f_dim in range(sub.shape[1]):
                for feature in np.unique(X[:,f_dim]):
                    self.cond_p_feature[i][(f_dim,feature)] = (np.sum(sub[:,f_dim] == feature) + alpha) / (sub.shape[0] + len(np.unique(X[:,f_dim])) * alpha)
                    
    def predict(self, test):
        p_data = {}
        for sub_label in np.unique(self.label):
            # 对概率值取log，防止乘积时浮点下溢
            p_data[sub_label] = np.log(self.pri_p_label[sub_label])
#            p_data[sub_label] = float(self.pri_p_label[sub_label])
#            p_data[sub_label] = self.pri_p_label[sub_label]
            for i in range(len(test)):
#                p_data[sub_label] *= np.log(self.cond_p_feature[sub_label][(i,test[i])])
                if self.cond_p_feature[sub_label].get((i,test[i])):
#                    p_data[sub_label] *= self.cond_p_feature[sub_label][(i,test[i])]
                    p_data[sub_label] += np.log(self.cond_p_feature[sub_label][(i,test[i])])
#                p_data[sub_label] *= self.cond_p_feature[sub_label][(i,test[i])]
#                print(sub_label, int(float(self.cond_p_feature[sub_label][(i,test[i])]) * 10))
#                p_data[sub_label] *= int(float(self.cond_p_feature[sub_label][(i,test[i])]) * 100)
        opt_label = max(p_data, key = p_data.get)
        return([opt_label, p_data.get(opt_label)])

model = MultinomialNB()
model.fit(X=X, y=label)
model.predict(test=z)

# =============================================================================
# 
# =============================================================================
#from sklearn import preprocessing

dataset = pd.read_csv("train.csv")

#le = preprocessing.LabelEncoder()
#for i in dataset.columns[1:]:
#    dataset[i] = le.fit_transform(dataset[i])
#dataset = np.array(dataset)
#sample_num = dataset.shape[0]


dataset = np.array(dataset)
dataset[:,1:][dataset[:,1:] != 0] = 1
sample_num = dataset.shape[0]

label = dataset[:,0]

# 分割训练集和测试集
train_dat, test_dat, train_label, test_label = train_test_split(dataset[:,1:], label, test_size = 0.2, random_state = 123456)

# 构建NB模型
model = MultinomialNB()
model.fit(X=train_dat, y=train_label, alpha=1)
# NB预测
pl = {}
i = 0
for test in test_dat:
    temp = model.predict(test=test)
    pl[i] = temp
    i += 1
# 输出测试错误率%
error = 0
for k,v in pl.items():
    if test_label[k] != v[0]:
        error += 1
print(error/len(test_label)*100)



#train_dat = dataset[:100,1:]
#train_label = dataset[:100,0]


#model2 = MultinomialNB()
#model2.fit(X=train_dat, y=train_label, alpha=1)
#model2.predict(test=train_dat[2])
#model2.pri_p_label
#model2.cond_p_feature

model.cond_p_feature[0][(122,0)]

train_dat = dataset[:int(sample_num*0.99),1:]
train_label = dataset[:int(sample_num*0.99),0]

test_dat = dataset[int(sample_num*0.99):,1:]
test_label = dataset[int(sample_num*0.99):,0]

test_dat = dataset[:int(sample_num*0.005),1:]
test_label = dataset[:int(sample_num*0.005),0]

model = MultinomialNB()
model.fit(X=train_dat, y=train_label, alpha=1)

model.predict(test=test_dat[2])

#test_dat = train_dat[:100,]
pl = {}
i = 0
for test in test_dat:
    temp = model.predict(test=test)
    pl[i] = temp
    i += 1

error = 0
for k,v in pl.items():
    if test_label[k] != v[0]:
        error += 1
print(error/len(test_label)*100)


pl = model.predict(test=test_dat)
model.pri_p_label
model.cond_p_feature

model.predict(test=test_dat[2])

p_data = {}
for sub_lable in np.unique(model.label):
    # 对概率值取log，防止乘积时浮点下溢
    p_data[sub_label] = np.log(model.pri_p_label[sub_label])
    for i in range(len(test)):
        p_data[sub_label] *= np.log(model.cond_p_feature[sub_label][(i,test[i])])
opt_label = max(p_data, key = p_data.get)


# =============================================================================
# 
# =============================================================================
dataset_voice = pd.read_csv("voice.csv")
dataset_voice[dataset_voice == 0] = np.nan
dataset_voice.fillna(dataset_voice.mean(), inplace=True)
dataset_voice = np.array(dataset_voice)
sample_num = dataset_voice.shape[0]

train_dat = dataset_voice[:int(sample_num*0.9),:-1]
train_label = dataset_voice[:int(sample_num*0.9),-1]

test_dat = dataset_voice[int(sample_num*0.9):,:-1]
test_label = dataset_voice[int(sample_num*0.9):,-1]

model = MultinomialNB()
model.fit(X=train_dat, y=train_label, alpha=1)


# =============================================================================
# 
# =============================================================================
from sklearn import preprocessing

dataset_crime = pd.read_csv("sf-crime.train.csv")
dataset_crime.head()
leCrime = preprocessing.LabelEncoder()
crime = leCrime.fit_transform(dataset_crime.Category)

days = pd.get_dummies(dataset_crime.DayOfWeek)
district = pd.get_dummies(dataset_crime.PdDistrict)
resolution = pd.get_dummies(dataset_crime.Resolution)

train_dat = pd.concat([days, district], axis=1)
train_dat = np.array(train_dat)
train_label = crime

train_dat = dataset_crime[dataset_crime.columns[3:6]]
train_dat = np.array(train_dat)
train_label = crime



model = MultinomialNB()
model.fit(X=train_dat, y=train_label, alpha=1)
#model.predict(test=train_dat[0])

test_dat = train_dat[:100,]
pl = {}
i = 0
for test in test_dat:
    temp = model.predict(test=test)
    pl[i] = temp
    i += 1

test_dat = train_dat[:100,]
pl2 = {}
i = 0
for test in test_dat:
    temp = model.predict(test=test)
#    temp.append(i)
    pl2[i] = temp
    i += 1

error = 0
for k,v in pl.items():
    if train_label[k] != v[0]:
        error += 1
print(error/100*100)

