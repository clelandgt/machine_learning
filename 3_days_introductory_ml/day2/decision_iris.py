# -*- coding: utf-8 -*-
# @File  : decision_iris.py
# @Author: clelandgt@163.com
# @Date  : 2019-11-11
# @Desc  : 使用决策树对鸢尾花进行分类
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def decision_iris():
    # 1. 获取数据
    iris = load_iris()

    # 2. 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

    # 3. 特征工程标准化
    transform = StandardScaler()
    x_train = transform.fit_transform(x_train)
    x_test = transform.fit_transform(x_test)

    # 4. 决策树
    estimator = DecisionTreeClassifier()
    estimator.fit(x_train, y_train)

    # 5. 模型评估
    # 方法一： 直接对比正式值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: \n", y_predict)
    print(u"直接对比真实值和预测值: \n", y_test==y_predict)

    # 方法儿: 计算准确度
    score = estimator.score(x_test, y_test)
    print(u"准确率： \n", score)


def main():
    decision_iris()


if __name__ == '__main__':
    main()
