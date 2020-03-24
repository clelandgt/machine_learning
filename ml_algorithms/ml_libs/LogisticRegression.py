# -*- coding: utf-8 -*-
# @File  : LogisticRegression.py
# @Author: clelandgt@163.com
# @Date  : 2020-03-24
# @Desc  : 逻辑回归的实现
import numpy as np
from sklearn import datasets


class LogisticRegression:
    def __init__(self):
        self._thetas = None
        # 截距
        self.intercept = None
        # 参数系数
        self.coefs = None

    @staticmethod
    def sigmoid(t):
        return 1.0 / (1 + np.exp(-t))

    def j(self, x, y, theta):
        """目标函数"""
        try:
            sig = self.sigmoid(x.dot(theta))
            return -(np.sum(y.dot(np.log(sig)) + (1-y).dot(1-sig))) / len(x)
        except:
            # 当数据特别大，报错时，返回无穷大。
            return float('inf')

    def dj_debug(self, x, y, theta, epsilon=0.01):
        """梯度调试"""
        res = np.empty(len(theta))
        for i in range(len(theta)):
            theta_1 = theta.copy()
            theta_1[i] += epsilon
            theta_2 = theta.copy()
            theta_2[i] -= epsilon
            res[i] = (self.j(theta_1, x, y) - self.j(theta_2, x, y)) / (2 * epsilon)
        return res

    def dj(self, x, y, thetas):
        """求导(梯度)"""
        sig = self.sigmoid(x.dot(thetas))
        return (x.T).dot(sig-y)

    def gradient_descent(self, dj, x, y, initial_thetas, eta, epsilon, max_iters):
        """梯度下降"""
        thetas = initial_thetas
        while max_iters > 0:
            # 梯度gradient
            gradient = dj(x, y, thetas)
            last_thetas = thetas
            thetas = thetas - eta * gradient
            if(abs(self.j(x, y, thetas) - self.j(x, y, last_thetas)) < epsilon):
                break
            max_iters -= 1

        self._thetas = thetas
        self.intercept = thetas[0]
        self.coefs = thetas[1:]

    def fit(self, x_train, y_train, eta=0.01, epsilon=1e-8, max_iters=1e4, debug=False):
        """训练"""
        # 加上一列全为1
        X_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_thetas = np.zeros(X_b.shape[1])

        if debug:
            self.gradient_descent(self.dj_debug, X_b, y_train, initial_thetas, eta, epsilon, max_iters)
        else:
            self.gradient_descent(self.dj, X_b, y_train, initial_thetas, eta, epsilon, max_iters)

    def predict(self, x_predict):
        X_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        p = self.sigmoid(X_b.dot(self._thetas))
        p[p >= 0.5] = 1
        p[p < 0.5] = 0

        return p

    def score(self, x, y):
        """accuracy=正确的数据量 / 样本量"""
        y_predict = self.predict(x)
        return np.sum(y == y_predict) / len(y)


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 只保留两种莺尾花
    X = X[y[y < 2], : 2]
    y = y[y < 2]

    # 训练模型
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    print('coefs: ', log_reg.coefs)
    print('intercept: ', log_reg.intercept)
    print('Score: ', log_reg.score(X, y))


if __name__ == '__main__':
    main()