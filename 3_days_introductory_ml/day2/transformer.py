# -*- coding: utf-8 -*-
# @File  : transformer.py
# @Author: clelandgt@163.com
# @Date  : 2019-11-07
# @Desc  :
from sklearn.preprocessing import StandardScaler


def transformer_demo():
    std = StandardScaler()
    data = [[1, 2, 3], [4, 5, 6]]
    print('fit_transform: \n', list(std.fit_transform(data)))

    std = StandardScaler()
    print('fit: \n', std.fit(data))
    print('transform: \n', std.transform(data))


def main():
    transformer_demo()


if __name__ == '__main__':
    main()