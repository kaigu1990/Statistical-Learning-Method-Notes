# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:16:00 2019

@author: gukai
"""

import numpy as np

A = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
#A = np.mat([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
B = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
#B = np.mat([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
PI = np.array([0.2,0.4,0.4])
#PI = np.mat([0.2,0.4,0.4])
V = ["red","white"]
Q = [1,2,3]
O = ["red", "white", "red"]

#Q_n = len(Q)
#O_n = len(O)
#alphas = np.zeros((O_n, Q_n))
#for t in range(O_n):
#    index = V.index(O[t])
#    if t == 0:
#        alphas[t] = PI * B[:,index]
#    else:
#        alphas[t] = np.dot(alphas[t-1], A) * B[:,index]
#P = np.sum(alphas[Q_n-1])

def forward(self, A, B, PI, Q, V, O):
    Q_n = len(Q)
    O_n = len(O)
    alphas = np.zeros((O_n, Q_n))
    for t in range(O_n):
        index = V.index(O[t])
        if t == 0:
            alphas[t] = PI * B[:,index]
        else:
            alphas[t] = np.dot(alphas[t-1], A) * B[:,index]
    P = np.sum(alphas[O_n-1])
    print("P(O|lambda)=", P, end="")
    return
# =============================================================================
# 
# =============================================================================
#Q_n = len(Q)
#O_n = len(O)
#betas = np.ones((O_n, Q_n))
#for t in range(O_n-2,-1,-1):
#    index = V.index(O[t+1])
#    betas[t] = np.dot(A * B[:,index], betas[t+1])
#P = np.dot(PI * B[:,V.index(O[0])], betas[0])

def backward(self, A, B, PI, Q, V, O):
    Q_n = len(Q)
    O_n = len(O)
    betas = np.ones((O_n, Q_n))
    for t in range(O_n-2,-1,-1):
        index = V.index(O[t+1])
        betas[t] = np.dot(A * B[:,index], betas[t+1])
    P = np.dot(PI * B[:,V.index(O[0])], betas[0])
    print("P(O|lambda)=", P, end="")
    return

# =============================================================================
# 
# =============================================================================
class Hidden_Markov_Model:
    def forward(self, A, B, PI, Q, V, O):
        Q_n = len(Q)
        O_n = len(O)
        alphas = np.zeros((O_n, Q_n))
        for t in range(O_n):
            index = V.index(O[t])
            if t == 0:
                alphas[t] = PI * B[:,index]
            else:
                alphas[t] = np.dot(alphas[t-1], A) * B[:,index]
        P = np.sum(alphas[O_n-1])
#        print("P(O|lambda)=", P, end="")
        return P

    def backward(self, A, B, PI, Q, V, O):
        Q_n = len(Q)
        O_n = len(O)
        betas = np.ones((O_n, Q_n))
        for t in range(O_n-2,-1,-1):
            index = V.index(O[t+1])
            betas[t] = np.dot(A * B[:,index], betas[t+1])
        P = np.dot(PI * B[:,V.index(O[0])], betas[0])
#        print("P(O|lambda)=", P, end="")
        return P

# =============================================================================
# 习题10.1
# =============================================================================
A = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
B = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
PI = np.array([0.2,0.4,0.4])
V = ["red","white"]
Q = [1,2,3]
O = ["red","white","red","white"]
        
HMM = Hidden_Markov_Model()
print("P(O|lambda) =", HMM.forward(A,B,PI,Q,V,O))
#HMM.forward(A,B,PI,Q,V,O)
#HMM.backward(A,B,PI,Q,V,O)
print("P(O|lambda) =", HMM.backward(A,B,PI,Q,V,O))

# =============================================================================
# 习题10.2
# =============================================================================

class Hidden_Markov_Model:
    def forward(self, A, B, PI, Q, V, O):
        Q_n = len(Q)
        O_n = len(O)
        alphas = np.zeros((O_n, Q_n))
        for t in range(O_n):
            index = V.index(O[t])
            if t == 0:
                alphas[t] = PI * B[:,index]
            else:
                alphas[t] = np.dot(alphas[t-1], A) * B[:,index]
        P = np.sum(alphas[O_n-1])
        print("P(O|lambda)=", P, end="")
        return alphas

    def backward(self, A, B, PI, Q, V, O):
        Q_n = len(Q)
        O_n = len(O)
        betas = np.ones((O_n, Q_n))
        for t in range(O_n-2,-1,-1):
            index = V.index(O[t+1])
            betas[t] = np.dot(A * B[:,index], betas[t+1])
        P = np.dot(PI * B[:,V.index(O[0])], betas[0])
        print("P(O|lambda)=", P, end="")
        return betas

A = np.array([[0.5,0.1,0.4],[0.3,0.5,0.2],[0.2,0.2,0.6]])
B = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
PI = np.array([0.2,0.3,0.5])
V = ["red","white"]
Q = [1,2,3]
O = ["red","white","red","red","white","red","white","white"]

HMM = Hidden_Markov_Model()
alpha = HMM.forward(A,B,PI,Q,V,O)
beta = HMM.backward(A,B,PI,Q,V,O)
p = alpha[3,2]*beta[3,2]/sum(alpha[3] * beta[3])
print("P(i4=q3|O,lambda) = ", p)