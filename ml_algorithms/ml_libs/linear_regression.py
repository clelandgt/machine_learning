# -*- coding: utf-8 -*-
# @File  : linear_regression.py
# @Author: clelandgt@163.com
# @Date  : 2020-03-22
# @Desc  : 线性回归实现
import numpy as np


class LinearRegressionGradientDescentCycle:
    def __init__(self):
        self._thetas = None
        # 截距
        self.intercept = None
        # 参数系数
        self.coefs = None

    def j(self, x, y, theta):
        """目标函数"""
        try:
            return np.sum(y - np.dot(x, theta)) ** 2 / len(x)
        except:
            # 当数据特别大，报错时，返回无穷大。
            return float('inf')

    def dj(self, x, y, thetas):
        """求导"""
        res = np.empty(len(thetas))
        # 第0个theta,其实就是截距
        res[0] = np.sum(np.dot(x, thetas) - y)
        for col in range(1, len(thetas)):
            res[col] = np.sum((np.dot(x, thetas) - y).dot(x[:, col]))

        return res * 2 / len(x)

    def gradient_descent(self, x, y, initial_thetas, eta=0.01, epsilon=1e-8, max_iters=1e4):
        """梯度下降"""
        thetas = initial_thetas
        while max_iters > 0:
            # 梯度gradient
            gradient = self.dj(x, y, thetas)
            last_thetas = thetas
            thetas = thetas - eta * gradient
            if(abs(self.j(x, y, thetas) - self.j(x, y, last_thetas)) < epsilon):
                break
            max_iters -= 1

        self._thetas = thetas
        self.intercept = thetas[0]
        self.coefs = thetas[1:]

    def fit(self, x_train, y_train, eta=0.01, epsilon=1e-8, max_iters=1e4):
        """训练"""
        # 将x转置加上一列全为1
        X_b = np.hstack([np.ones((len(x_train), 1)), x_train.reshape(-1, 1)])
        initial_thetas = np.zeros(X_b.shape[1])
        self.gradient_descent(X_b, y_train, initial_thetas, eta, epsilon, max_iters)

    def predict(self, x_predict):
        """预测"""
        X_b = np.hstack([np.ones((len(x_predict), 1)), x_predict.reshape(-1, 1)])
        return np.dot(X_b, self._thetas)

    @staticmethod
    def score(y, y_predict):
        """R评价"""
        return 1 - np.dot(y_predict - y, y_predict - y) / len(y) / np.var(y)


def main():

    # 1. 初始化数据
    num_size = 200
    np.random.seed(100)
    # np.random.random(size=num_size) 0-1的随机数；np.random.normal(size=num_size) 正态分布
    x = np.random.random(size=num_size)
    # 函数 y=10x+5 然后设置一定的随机波动
    y = 10 * x + 5 + np.random.normal(size=num_size)

    linear = LinearRegressionGradientDescentCycle()
    linear.fit(x, y)
    predict_y = linear.predict(x)
    print('coefs: ', linear.coefs)
    print('intercept: ', linear.intercept)
    print('score: ', linear.score(y, predict_y))


if __name__ == '__main__':
    main()
