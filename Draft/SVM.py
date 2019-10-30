# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:35:52 2019

@author: kgu
"""

import numpy as np
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split

dataset = pd.read_table("testSet.txt", header = None)
dataset = np.mat(dataset)
X = dataset[:,:-1]
y = dataset[:,-1]
y[y == 0] = -1

#sigma = 10
#b = 0
#C = 200
#numSamples = X.shape[0]
#
#alphas = np.mat(np.zeros((numSamples,1)))
#E = np.mat(np.zeros((numSamples, 1)))
#toler = 0.001

class SVM:
    def __init__(self, X, y, sigma = 10, C = 200, toler = 0.001):
        self.X = X
        self.y = np.ravel(y)
        self.numSamples = X.shape[0]
        self.sigma = sigma
        self.C = C
        self.toler = toler
        
        self.b = 0
        self.alphas = np.zeros(self.X.shape[0])
#        self.alphas = [0] * self.X.shape[0]
#        self.alphas = np.mat(np.zeros((X.shape[0],1)))
#        self.E = [0] * self.X.shape[0]
        self.E = np.zeros(self.X.shape[0])
        self.kernelMatrix = self.calc_Kernel()


    def calc_Kernel(self):
        k_matrix = np.mat(np.zeros((self.numSamples, self.numSamples)))
        for i in range(self.numSamples):
            x = self.X[i,:]
            for j in range(i, self.numSamples):
                z = self.X[j,:]
                r = (x - z) * (x - z).T
                gsi = np.exp(-1 * r / (2 * self.sigma ** 2))
                k_matrix[i,j] = gsi
                k_matrix[j,i] = gsi
        return k_matrix

    def calc_E(self, i):
        gx = np.multiply(self.alphas, self.y) * self.kernelMatrix[:,i] + self.b
        Ei = gx - self.y[i]
#        gx = 0
#        index = [i for i, alpha in enumerate(self.alphas) if alpha != 0]
#        
#        for j in index:
#            gx += self.alphas[j] * self.y[j] * self.kernelMatrix[j,i]
#            
#        Ei = gx + self.b - self.y[i]  
        
        return Ei
#
#    def select_AlphaJ(self, E1_index, E1):
#        max_diff = -1
#        E2 = 0
#        E2_index = -1
#        nozeroE = [i for i, E_value in enumerate(self.E) if E_value != 0]
#        if (len(nozeroE) > 0):
#            for j in nozeroE:
#                Ej = self.calc_E(j)
#                if (math.fabs(E1 - Ej) > max_diff):
#                    max_diff = math.fabs(E1 - Ej)
#                    E2_index = j
#                    E2 = Ej      
#        else:
#            E2_index = E1_index
#            while E2_index == E1_index:
#                E2_index = int(random.uniform(0, self.numSamples))
#            E2 = self.calc_E(E2_index)
#            
#        return E2_index, E2
        
    def select_AlphaJ(self, i, E1):
        E2 = 0
        maxE1_E2 = -1
        maxIndex = -1
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]

        if (len(nozeroE) > 0):
            for j in nozeroE:
                if j == i: continue
                Ej = self.calc_E(j)
                if math.fabs(E1 - Ej) > maxE1_E2:
                    maxE1_E2 = math.fabs(E1 - Ej)
                    E2 = Ej
                    maxIndex = j
        else:
            maxIndex = i
            while maxIndex == i:
                maxIndex = int(random.uniform(0, self.numSamples))
            E2 = self.calc_E(maxIndex)
         
#        for j in nozeroE:
#            E2_tmp = self.calc_E(j)
#            if math.fabs(E1 - E2_tmp) > maxE1_E2:
#                maxE1_E2 = math.fabs(E1 - E2_tmp)
#                E2 = E2_tmp
#                maxIndex = j
#        #如果列表中没有非0元素了（对应程序最开始运行时的情况）
#        if maxIndex == -1:
#            maxIndex = i
#            while maxIndex == i:
#                maxIndex = int(random.uniform(0, self.numSamples))
#            E2 = self.calc_E(maxIndex)

        return maxIndex, E2

    def Loop(self, i):
        E1 = self.calc_E(i)
        if (self.y[i] * E1 < -self.toler and self.alphas[i] < self.C) or (self.y[i] * E1 > self.toler and self.alphas[i] > 0):
            j,E2 = self.select_AlphaJ(i, E1)
            alpha_old_1 = self.alphas[i]
            alpha_old_2 = self.alphas[j]
            y1 = self.y[i]
            y2 = self.y[j]
            if y1 != y2:
                L = max(0, alpha_old_2 - alpha_old_1)
                H = min(self.C, self.C + alpha_old_2 - alpha_old_1)
            else:
                L = max(0, alpha_old_2 + alpha_old_1 - self.C)
                H = min(self.C, alpha_old_2 + alpha_old_1)
                
            if L == H:
                return 0
                
            alpha_new_2 = alpha_old_2 + y2 * (E1 - E2) / (self.kernelMatrix[i,i] + self.kernelMatrix[j,j] - 2 * self.kernelMatrix[i,j])
                
            if (alpha_new_2 > H):
                alpha_new_2 = H
            elif (alpha_new_2 < L):
                alpha_new_2 = L
                            
            alpha_new_1 = alpha_old_1 + y1 * y2 * (alpha_old_2 - alpha_new_2)
                
            b1_new = -1 * E1 - y1 * self.kernelMatrix[i,i] * (alpha_new_1 - alpha_old_1) - y2 * self.kernelMatrix[j,i] * (alpha_new_2 - alpha_old_2) + self.b
            b2_new = -1 * E2 - y1 * self.kernelMatrix[i,j] * (alpha_new_1 - alpha_old_1) - y2 * self.kernelMatrix[j,j] * (alpha_new_2 - alpha_old_2) + self.b
                        
            if (alpha_new_1 > 0) and (alpha_new_1 < self.C):
                b_new = b1_new
            elif (alpha_new_2 > 0) and (alpha_new_2 < self.C):
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2
                    
            self.alphas[i] = alpha_new_1
            self.alphas[j] = alpha_new_2
            self.b = b_new
            self.E[i] = self.calc_E(i)
            self.E[j] = self.calc_E(j)
                
            if (math.fabs(alpha_new_2 - alpha_old_2) < 0.00001):
                return 0
            else:
                return 1
            
        else:
            return 0

    def train(self, max_iter = 1000):
        alpha_change = 1
        iter = 0
        entireSet = True
        
        while (iter < max_iter) and (alpha_change > 0) or entireSet:
            alpha_change = 0
            iter += 1
            if entireSet:
                for i in range(self.numSamples):
                    alpha_change += self.Loop(i)
                    print("iter:", iter, "i value:", i, "entire_Set, alpha changed:", alpha_change, "b:", self.b)
            else:
                support_vector_i = [i for i, alpha in enumerate(self.alphas) if alpha > 0 * alpha < self.C]
                for i in support_vector_i:
                    alpha_change += self.Loop(i)
                    print("iter:", iter, "i value:", i, "support_vector_Set, alpha changed:", alpha_change, "b:", self.b)
                    
            if entireSet:
                entireSet = False
            elif alpha_change == 0:
                entireSet = True

    def calcKernelValue(self, xj, xi):
        r = (xj - xi) * (xj - xi).T
        gsi = np.exp(-1 * r / (2 * self.sigma ** 2))
        
        return gsi
    
    def predict(self, test_X):
#        n = test_X.shape[0]
        support_vector = np.nonzero(self.alphas > 0)[0]
        gx = 0
        for i in support_vector:
            kernel_value = self.calcKernelValue(self.X[i,:], test_X)
            gx += self.alphas[i] * self.y[i] * kernel_value
        gx += self.b
        return np.sign(gx)

        
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 123456)


svm = SVM(X=train_X, y=train_y, C=10)
svm.train()

error = 0
for i in range(test_X.shape[0]):
    p = svm.predict(test_X[i])
    if p != test_y[i]:
        error += 1

print(error / len(test_y) * 100)

# =============================================================================
# 
# =============================================================================


svm2 = SVM(X=train_X, y=train_y)


alpha_change = 1
iter = 0
max_iter = 200

while (iter < max_iter) and (alpha_change > 0):
    alpha_change = 0
    iter += 1
    for i in range(svm2.numSamples):
        E1 = svm2.calc_E(i)
        if (svm2.y[i] * E1 < -svm2.toler and svm2.alphas[i] < svm2.C) or (svm2.y[i] * E1 > svm2.toler and svm2.alphas[i] > 0):
            j,E2 = svm2.getAlphaJ(i, E1)
            
            alpha_old_1 = svm2.alphas[i]
            alpha_old_2 = svm2.alphas[j]
            
            y1 = svm2.y[i]
            y2 = svm2.y[j]
            
            if y1 != y2:
                L = max(0, alpha_old_2 - alpha_old_1)
                H = min(svm2.C, svm2.C + alpha_old_2 - alpha_old_1)
            else:
                L = max(0, alpha_old_2 + alpha_old_1 - svm2.C)
                H = min(svm2.C, alpha_old_2 + alpha_old_1)
                
            if L == H:
                continue
            
            k11 = svm2.kernelMatrix[i,i]
            k22 = svm2.kernelMatrix[j,j]
            k21 = svm2.kernelMatrix[j,i]
            k12 = svm2.kernelMatrix[i,j]
            
            alpha_new_2 = alpha_old_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)
#            alpha_new_2 = alpha_old_2 + y2 * (E1 - E2) / (svm2.kernelMatrix[i,i] + svm2.kernelMatrix[j,j] - 2 * svm2.kernelMatrix[i,j])
                
            if (alpha_new_2 > H):
                alpha_new_2 = H
            elif (alpha_new_2 < L):
                alpha_new_2 = L
            
            alpha_new_1 = alpha_old_1 + y1 * y2 * (alpha_old_2 - alpha_new_2)
#            alpha_new_1 = alpha_old_1 + y1 * y2 * (alpha_old_2 - alpha_new_2)
            
            b1_new = -1 * E1 - y1 * k11 * (alpha_new_1 - alpha_old_1) - y2 * k21 * (alpha_new_2 - alpha_old_2) + svm2.b
            b2_new = -1 * E2 - y1 * k12 * (alpha_new_1 - alpha_old_1) - y2 * k22 * (alpha_new_2 - alpha_old_2) + svm2.b
            
#            b1_new = -1 * E1 - y1 * svm2.kernelMatrix[i,i] * (alpha_new_1 - alpha_old_1) - y2 * svm2.kernelMatrix[j,i] * (alpha_new_2 - alpha_old_2) + svm2.b
#            b2_new = -1 * E2 - y1 * svm2.kernelMatrix[i,j] * (alpha_new_1 - alpha_old_1) - y2 * svm2.kernelMatrix[j,j] * (alpha_new_2 - alpha_old_2) + svm2.b
                        
            if (alpha_new_1 > 0) and (alpha_new_1 < svm2.C):
                b_new = b1_new
            elif (alpha_new_2 > 0) and (alpha_new_2 < svm2.C):
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2
                    
            svm2.alphas[i] = alpha_new_1
            svm2.alphas[j] = alpha_new_2
            svm2.b = b_new
            svm2.E[i] = svm2.calc_E(i)
            svm2.E[j] = svm2.calc_E(j)
                
            if (math.fabs(alpha_new_2 - alpha_old_2) >= 0.00001):
                alpha_change += 1
        print("iter:", iter, "i value:", i, "entire_Set, alpha changed:", alpha_change, "b:", svm2.b)





                        

                
    



