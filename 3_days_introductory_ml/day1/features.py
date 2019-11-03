# -*- coding: utf-8 -*-
# @File  : features.py
# @Author: clelandgt@163.com
# @Date  : 2019-11-03
# @Desc  : day1学习代码
import jieba
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def datasets_demo():
    # 加载数据
    iris = load_iris()
    print(u"***************鸢尾花数据集的返回***************")
    print(u"类型: {0}\n 返回值:  \n {1}", type(iris), iris)
    print(u"数据描述: \n{}".format(iris.DESCR))
    print(u"特征值: \n{}".format(iris.data))
    print(u"目标值: \n{}".format(iris.target))
    print(u"目标名字: \n{}".format(iris.target_names))

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print(u"训练集特征值:\n", x_train.data, x_train.shape)


def dict_feature_extract_demo():
    data = [
        {
            'city': u'北京',
            'temperature': 100
        },
        {
            'city': u'上海',
            'temperature': 60
        },
        {
            'city': u'深圳',
            'temperature': 30
        },
    ]
    # 实例化一个转化器，默认使用稀疏矩阵: sparse=True
    transform = DictVectorizer()

    # 转化特征值
    data_new = transform.fit_transform(data)
    print("data new: \n", data_new)
    print(u"特征名称: \n", transform.get_feature_names())

    # 不转化为稀疏矩阵
    transform = DictVectorizer(sparse=False)
    data_new = transform.fit_transform(data)
    print(u"非稀疏矩阵")
    print("data new: \n", data_new)
    print(u"特征名称: \n", transform.get_feature_names())


def text_feature_extract_demo():
    """
    文本特征提取: CountVectorizer
    :return:
    """
    # 1. 实例化转化器类, stop_words参数加入需排除的分词
    data = ['life is shot, i like like python', 'life is too long, i dislike python']
    transform = CountVectorizer(stop_words=['is', 'too'])

    # 2. 文本特征提取，
    data_new = transform.fit_transform(data)

    # 使用toarray()将onehot转化为二维数组
    print('data_new: \n', data_new.toarray())
    print(u'特征名称: \n', transform.get_feature_names())


def chinese_text_feature_extract_demo():
    """
    中文文本的手动分词, 然后提取
    文本：['我爱北京天安门', '天安门上太阳升']
    :return:
    """
    # 1. 实例化转化器类，手动使用空格将中文的字和短语分开
    data = [u'我 爱 北京 天安门', u'天安门 上 太阳 升']
    transform = CountVectorizer()

    # 2. 文本特征提取
    data_new = transform.fit_transform(data)

    # 使用toarray()将onehot转化为二维数组
    print('data_new: \n', data_new.toarray())
    print(u'特征名称: \n', transform.get_feature_names())


def chinese_text_extract_with_jieba_demo():
    """
    使用结巴对中文文本实现自动分词，然后提取

    文本：['我爱北京天安门', '天安门上太阳升']
    :return:
    """
    # 1. 实例化转化器
    data = [u'我爱北京天安门', '天安门上太阳升']
    data_new = []
    for item in data:
        data_new.append(cut_word(item))
    print(u'结巴分词后: \n', data_new)
    transform = CountVectorizer()

    # 文本提取
    result_data = transform.fit_transform(data_new)
    print('data_new: \n', result_data.toarray())
    print(u'特征名称: \n', transform.get_feature_names())


def chinese_text_extract_tfidf_demo():
    """
    tfidf
    :return:
    """
    # 1. 实例化转化器
    data = [u'某个单词或短语在一篇文章中出现的频率高，并且在其他文章很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类',
            u'评估一字词对于一个文件集或者一个语料库中的其中一份文件的重要程度',
            u'逆向文档概率(inverse document frequency, idf) 词语的普遍重要程度。总文件数目/包含该词语的文件数目，再将所得值取10为底的对数得到。']
    data_new = []
    for item in data:
        data_new.append(cut_word(item))
    print(u'结巴分词后: \n', data_new)
    transform = TfidfVectorizer()

    # 文本提取
    result_data = transform.fit_transform(data_new)
    print('data_new: \n', result_data.toarray())
    print(u'特征名称: \n', transform.get_feature_names())


def cut_word(text):
    """
    结巴分词
    :param text:
    :return:
    """
    return ' '.join(jieba.cut(text))


def minmax_demo():
    """
    归一化
    :return:
    """
    # 1. 获取数据
    df = pd.read_csv('statics/dating.txt')
    df = df[['milage', 'Liters', 'Consumtime']]
    print(df)

    # 2. 实例化转换器
    transform = MinMaxScaler()

    # 3. 调用fit_transform
    result = transform.fit_transform(df)
    print('result: \n', result)


def main():
    """
    主函数
    :return:
    """
    # 数据集使用
    # datasets_demo()

    # 字典特征提取
    # dict_feature_extract_demo()

    # 英文特征提取
    # text_feature_extract_demo()

    # 中文特征提取(手动分词)
    # chinese_text_feature_extract_demo()

    # 中文特征提取(自动分词)
    # chinese_text_extract_with_jieba_demo()

    # tfidf
    # chinese_text_extract_tfidf_demo()

    # 归一化处理
    minmax_demo()



if __name__ == '__main__':
    main()
