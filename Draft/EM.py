import numpy as np
import math
import random
import copy

# 公式9.25，计算高斯分布密度
def gaussian_pro(x, mu, sigma):
    gsi = np.exp(-1 * (x - mu) * (x - mu) / (2 * sigma**2)) * \
    (1 / (math.sqrt(2 * math.pi) * sigma))
    return(gsi)

# 算法9.2的E步，计算分模型k对观测数据yi的响应度gamma
def E_step(x, alpha_s, mu_s, sigma_s, K):
    gamma = []
    for i in range(K): 
        gamma.append(alpha_s[i] * gaussian_pro(x, mu_s[i], sigma_s[i]))
    
    sum_gamma = sum(gamma)
    
    for i in range(K): 
        gamma[i] /= sum_gamma
    return(gamma)

# 算法9.2的M步，更新模型参数mu,sigma,alpha
def M_step(x, mu_s, gamma, K):
    mu_k = []
    sigma_k = []
    alpha_k = []
    
    for i in range(K): 
        # 更新 mu
        mu_k.append(np.dot(gamma[i], x) / np.sum(gamma[i]))
        # 更新 sigam
        sigma_k.append(math.sqrt(np.dot(gamma[i], (x - mu_s[i])**2) / np.sum(gamma[i])))
        # 更新 alpha
        alpha_k.append(np.sum(gamma[i]) / len(gamma[i]))
    return mu_k, sigma_k, alpha_k

# 迭代E步和M步，直至收敛
def EM(x, K, max_iter = 500):
    # 初始化alpha, mu, sigma
    alpha = [1 / K for i in range(K)]
    random.seed(12345)
    mu = [random.random() for i in range(K)]
    sigma = [random.random()*10 for i in range(K)]
    
    gamma = np.zeros((K, len(x)))
    iter = 0
    while iter < max_iter:
        gamma_old = copy.deepcopy(gamma)
        gamma = E_step(x, alpha, mu, sigma, K)
        mu, sigma, alpha = M_step(x, mu, gamma, K)
        # gamma变化小于1e-10则停止迭代
        if np.sum(abs(gamma[0] - gamma_old[0])) < 1e-10:
            break
        else:
            iter += 1
    return alpha, mu, sigma


# 按照习题例子输入观测数据（这里是简单的一维数据），通过EM函数估计两个分量的高斯混合模型的参数
alpha, mu, sigma = EM(dataset, 2)
print("alpha1:%.3f, alpha2:%.3f" %(alpha[0], alpha[1]))
print("mu1:%.1f, mu2:%.1f" %(mu[0], mu[1]))
print("sigma1:%.1f, sigma2:%.1f" %(sigma[0], sigma[1]))
# alpha1:0.658, alpha2:0.342
# mu1:21.6, mu2:19.7
# sigma1:44.5, sigma2:8.3





