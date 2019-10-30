import numpy as np
import pandas as pd

class Adaboost:
    def calc_e(self, X, y, i, threshold, orientation, D):
        # 计算错误率和G(x)分类结果
        e = np.ones((X.shape[0],1))
        Gx = np.zeros((X.shape[0],1))
        if orientation == "left":
            Gx[X[:,i] <= threshold] = 1
            Gx[X[:,i] > threshold] = -1
        else:
            Gx[X[:,i] > threshold] = 1
            Gx[X[:,i] <= threshold] = -1
        e[Gx == y] = 0
        # 加权误差weight_e
        weight_e = D * e
        return weight_e, Gx

    def build_stump(self, X, y, D):
        # 设置步长，对于非二值化的数据而言
        numSteps = 4
        # 用于存储决策树桩的一些数据，比如切点、分类方向、加权误差等
        best_stump = {}
        weight_e_min = 1
        
        for i in range(X.shape[1]):
            X_min = X[:,i].min()
            X_max = X[:,i].max()
            step_size = (X_max-X_min) / numSteps
            for j in range(-1, int(numSteps) + 1):
                for ori in ["left", "right"]:
                    thr = X_min + j * step_size
                    weight_e, Gx = self.calc_e(X, y, i, thr, ori, D)
                    if weight_e < weight_e_min:
                        weight_e_min = weight_e
                        best_stump["e"] = weight_e_min
                        best_stump["threshold"] = thr
                        best_stump["orientation"] = ori
                        best_stump["Gx"] = Gx
                        best_stump["feature"] = i
        return best_stump

    def adaboost_classfier(self, X, y, max_iter = 200):
        m, n = np.shape(X)
        # 初始化样本权值
        D = np.mat([1 / m] * m)
        # 存储每个弱分类器
        weak_classfier = []
        fx = np.mat(np.zeros((m,1)))
        
        for i in range(max_iter):
            stump = self.build_stump(X, y, D)
            Gx = stump["Gx"]
            
            # 公式8.2，计算alpha
            alpha = 1/2 * np.log((1 - stump["e"]) / stump["e"])
            
            # 公式8.4，更新样本权值
            D = np.multiply(D, np.exp(-1 * alpha * np.multiply(y, stump["Gx"]).T))
            D = D / np.sum(D)
            
            stump["alpha"] = alpha
            weak_classfier.append(stump)
    
            # 构建线性组合分类器
            fx += np.multiply(alpha, Gx)
            # 计算测试误差，为0则结束迭代
            error = np.sum(np.sign(fx) != y)
            if error == 0: break
        return weak_classfier


    def predict(self, X_test, classfier):
        for stump in classfier:
            threshold = stump["threshold"]
            orientation = stump["orientation"]
            feature = stump["feature"]
            
            if orientation == "left":
                if X_test[:,feature] <= threshold:
                    return 1
                else:
                    return -1
            else:
                if X_test[:,feature] > threshold:
                    return 1
                else:
                    return -1




train_X = np.mat([[0.,1.,3.],
                  [0.,3.,1.],
                  [1.,2.,2.],
                  [1.,1.,3.],
                  [1.,2.,3.],
                  [0.,1.,2.],
                  [1.,1.,2.],
                  [1.,1.,1.],
                  [1.,3.,1.],
                  [0.,2.,1.]])
train_y = np.mat([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]).T
adaboost = Adaboost()
tree = adaboost.adaboost_classfier(X=train_X, y=train_y, max_iter = 50)
len(tree)



