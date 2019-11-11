# -*- coding: utf-8 -*-
# @File  : nb_news.py
# @Author: clelandgt@163.com
# @Date  : 2019-11-10
# @Desc  :

""" 20个新闻组分类
20个新闻组数据大约20000个新闻组文档的集合，平均分布在20个不同的新闻组中，据我所知，它最初有Ken Lang收集的。这20个新闻组集合已经成为机器学习
技术的文本应用中实验流行的数据集，例如文本分类和聚类。

"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def nb_news():
    """
    朴素贝叶斯算法对新闻分类
    :return:
    """
    # 1. 获取数据
    news = fetch_20newsgroups(subset='all')
    print(news.data)

    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # 3. 特征工程：文本抽取-tfidf
    transform = TfidfVectorizer()
    x_train = transform.fit_transform(x_train)
    y_train = transform.transform(y_train)

    # 4. 朴树贝叶斯
    estimator = MultinomialNB()
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
    nb_news()


if __name__ == '__main__':
    main()

