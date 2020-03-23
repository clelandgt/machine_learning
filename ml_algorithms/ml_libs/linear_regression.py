# -*- coding: utf-8 -*-
# @File  : linear_regression.py
# @Author: clelandgt@163.com
# @Date  : 2020-03-22
# @Desc  : 线性回归的多种实现
import numpy as np


class LinearRegression:
    def __init__(self):
        self._thetas = None
        # 截距
        self.intercept = None
        # 参数系数
        self.coefs = None

    def fit(self, x_train, y_train):
        raise NotImplementedError

    def predict(self, x_predict):
        raise NotImplementedError

    @staticmethod
    def score(y, y_predict):
        """R评价"""
        return 1 - np.dot(y_predict - y, y_predict - y) / len(y) / np.var(y)


class LinearRegressionBGDLoop(LinearRegression):
    """ 批量梯度下降(BGD)_循环维度 """
    def j(self, x, y, theta):
        """目标函数"""
        try:
            return np.sum(y - np.dot(x, theta)) ** 2 / len(x)
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
        res = np.empty(len(thetas))
        # 第0个theta,其实就是截距
        res[0] = np.sum(np.dot(x, thetas) - y)
        for col in range(1, len(thetas)):
            res[col] = np.sum((np.dot(x, thetas) - y).dot(x[:, col]))

        return res * 2 / len(x)

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
        """预测"""
        X_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        return np.dot(X_b, self._thetas)

    @staticmethod
    def score(y, y_predict):
        """R评价"""
        return 1 - np.dot(y_predict - y, y_predict - y) / len(y) / np.var(y)


class LinearRegressionBGDVector(LinearRegressionBGDLoop):
    """批量梯度下降(BGD)_向量化维度"""
    def dj(self, x, y, thetas):
        """向量实现"""
        return (x.T).dot(x.dot(thetas)-y) * 2 / len(x)


class LinearRegressionSGD(LinearRegressionBGDLoop):
    """随机梯度下降(SDG)"""
    def dj(self, x_i, y_i, thetas):
        return (x_i.T).dot(x_i.dot(thetas)-y_i) * 2

    def gradient_descent(self, dj, x, y, initial_thetas, eta, epsilon, max_iters, t0, t1):
        """梯度下降"""
        thetas = initial_thetas
        while max_iters > 0:
            # 随机抽取一个样本
            i = np.random.randint(len(x))
            # 梯度gradient
            gradient = dj(x[i], y[i], thetas)
            last_thetas = thetas
            # 学习率模拟退火
            eta = (t0 + eta) / (max_iters + t1)
            thetas = thetas - eta * gradient
            if(abs(self.j(x, y, thetas) - self.j(x, y, last_thetas)) < epsilon):
                break
            max_iters -= 1

        self._thetas = thetas
        self.intercept = thetas[0]
        self.coefs = thetas[1:]

    def fit(self, x_train, y_train, eta=0.01, epsilon=1e-8, max_iters=1e4, t0=5, t1=50, debug=False):
        """训练"""
        # 加上一列全为1
        X_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_thetas = np.zeros(X_b.shape[1])

        if debug:
            self.gradient_descent(self.dj_debug, X_b, y_train, initial_thetas, eta, epsilon, max_iters, t0, t1)
        else:
            self.gradient_descent(self.dj, X_b, y_train, initial_thetas, eta, epsilon, max_iters, t0, t1)


def main():
    # 1. 初始化数据
    num_size = 2000
    np.random.seed(100)
    # np.random.random(size=num_size) 0-1的随机数；np.random.normal(size=num_size) 正态分布
    x = np.random.random(size=num_size)
    # 函数 y=10x+5 然后设置一定的随机波动
    y = 10 * x + 5 + np.random.normal(size=num_size)
    x = x.reshape(-1, 1)

    lin_reg = LinearRegressionBGDLoop()
    lin_reg.fit(x, y)
    predict_y = lin_reg.predict(x)
    print('---- BGD Loop ----')
    print('coefs: ', lin_reg.coefs)
    print('intercept: ', lin_reg.intercept)
    print('score: ', lin_reg.score(y, predict_y))

    lin_reg = LinearRegressionBGDVector()
    lin_reg.fit(x, y)
    predict_y = lin_reg.predict(x)
    print('---- BGD Vector ----')
    print('coefs: ', lin_reg.coefs)
    print('intercept: ', lin_reg.intercept)
    print('score: ', lin_reg.score(y, predict_y))

    lin_reg = LinearRegressionSGD()
    lin_reg.fit(x, y)
    predict_y = lin_reg.predict(x)
    print('---- SGD ----')
    print('coefs: ', lin_reg.coefs)
    print('intercept: ', lin_reg.intercept)
    print('score: ', lin_reg.score(y, predict_y))


if __name__ == '__main__':
    main()
