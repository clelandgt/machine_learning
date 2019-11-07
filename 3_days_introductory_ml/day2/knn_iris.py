# -*- coding: utf-8 -*-
# @File  : knn_iris.py
# @Author: clelandgt@163.com
# @Date  : 2019-11-07
# @Desc  :
""" iris数据集介绍
iris数据集时常用分类实验数据集，由Fisher 1936年搜集整理。Iris也称鸢尾花数据集，是一类多重变量分析的数据集。数据集介绍：
属性数量：4(数值型，数值型，帮助预测的属性和类)
实例数量：150(三类各有50个)
    - sepal length 萼片长度(厘米)
    - sepal width 萼片宽度(厘米)
    - petal length 花瓣长度(厘米)
    - petal width 花瓣宽度(厘米)

类别：
    - lris-Setosa 山鸢尾
    - lris-Vericolour 变色鸢尾
    - lris-Virginca 维吉尼亚鸢尾
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def knn_iris():
    """
    使用KNN算法对鸢尾花进行分类
    :return:
    """
    # 1.获取数据
    iris = load_iris()

    # 2.数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
    print("Data: \n", x_train)

    # 3.特征工程：标准化
    transformer = StandardScaler()
    train_data = transformer.fit_transform(x_train)
    test_data = transformer.fit_transform(x_test)
    print("Preprocessing: \n", train_data)

    # 4.KNN算法预估器
    estimator = KNeighborsClassifier()
    estimator.fit(train_data, y_train)
    print("Predict: \n", estimator.predict(train_data))
    print("Target: \n", y_train)

    # 5.模型评估
    print('Train score: \n', estimator.score(train_data, y_train))
    print('Test score: \n', estimator.score(test_data, y_test))


def knn_iris_grid():
    """
    使用KNN算法对鸢尾花进行分类. 调优方法使用：网格搜索
    :return:
    """
    # 1.获取数据
    iris = load_iris()

    # 2.数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
    print("Data: \n", x_train)

    # 3.特征工程：标准化
    transformer = StandardScaler()
    train_data = transformer.fit_transform(x_train)
    test_data = transformer.fit_transform(x_test)
    print("Preprocessing: \n", train_data)

    # 4.KNN算法预估器
    estimator = KNeighborsClassifier()
    param_dict = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(train_data, y_train)

    # 5.模型评估
    print('Train score: \n', estimator.score(train_data, y_train))
    print('Test score: \n', estimator.score(test_data, y_test))

    # 最佳参数
    print(u'最佳参数: \n', estimator.best_params_)
    # 最佳结果
    print(u'结果: \n', estimator.best_score_)
    # 最佳估计器
    print(u'估计器: \n', estimator.best_estimator_)
    # 交叉验证结果
    print(u'估计器: \n', estimator.cv_results_)


def main():
    # KNN算法对鸢尾花进行分类
    # knn_iris()

    # KNN算法对鸢尾花进行分类，模型优化
    knn_iris_grid()


if __name__ == '__main__':
    main()
