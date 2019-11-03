# -*- coding: utf-8 -*-
# @File  : features.py
# @Author: clelandgt@163.com
# @Date  : 2019-11-03
# @Desc  : 鸢尾花

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


def datasets_demo():
    # 加载数据
    iris = load_iris()
    print("***************鸢尾花数据集的返回***************")
    print("类型: {0}\n 返回值:  \n {1}", type(iris), iris)
    print("数据描述: \n{}".format(iris.DESCR))
    print("特征值: \n{}".format(iris.data))
    print("目标值: \n{}".format(iris.target))
    print("目标名字: \n{}".format(iris.target_names))

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集特征值:\n", x_train.data, x_train.shape)


def dict_feature_extract_demo():
    data = [
        {
            'city': '北京',
            'temperature': 100
        },
        {
            'city': '上海',
            'temperature': 60
        },
        {
            'city': '深圳',
            'temperature': 30
        },
    ]
    # 实例化一个转化器，默认使用稀疏矩阵: sparse=True
    transform = DictVectorizer()

    # 转化特征值
    data_new = transform.fit_transform(data)
    print("data new: \n", data_new)
    print("特征名称: \n", transform.get_feature_names())

    # 不转化为稀疏矩阵
    transform = DictVectorizer(sparse=False)
    data_new = transform.fit_transform(data)
    print("非稀疏矩阵")
    print("data new: \n", data_new)
    print("特征名称: \n", transform.get_feature_names())


def main():
    # 数据集使用
    # datasets_demo()

    # 特征值提取
    dict_feature_extract_demo()



if __name__ == '__main__':
    main()
