#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework4.py

    DESCRIPTION
        1. 读入ag0613.csv数据集，并计算数据的最大值、最小值、均值、标准差、中位数
        2. 矩阵计算
            np.mat([[1, 2], [3, 4]]) * np.mat([[2, 5], [1, 3]])
        3. 随机生成100个标准正态的数据，并计算数据的均值与标准差

    MODIFIED  (MM/DD/YY)
        Na  11/14/2018

"""
__VERSION__ = "1.0.0.11142018"


# imports
import os
import numpy as np
import matplotlib.pyplot as plt

# consts
CUR_PATH = os.path.dirname(os.path.abspath(__file__))

# functions
def main():
    # 1. 读入ag0613.csv数据集，并计算数据的最大值、最小值、均值、标准差、中位数
    print('读入ag0613.csv数据集...')
    data_file_fullpath = os.path.join(CUR_PATH, 'ag0613.csv')
    data = np.loadtxt(data_file_fullpath, dtype=np.int16, skiprows=1)
    dmax, dmin, dmean = data.max(), data.min(), data.mean()
    dstd, dmedian = data.std(), np.median(data)
    print('数据的最大值: {0}\n{1}最小值: {2}\n{1}均值:   {3}\n{1}标准差: {4}\n{1}中位数: {5}'.
          format(dmax, ' ' * 6, dmin, dmean, dstd, dmedian))

    # 2. 矩阵计算
    m1 = np.mat([[1, 2], [3, 4]])
    m2 = np.mat([[2, 5], [1, 3]])
    result = m1.dot(m2)
    print('\n矩阵1\n{0}\n点乘矩阵2\n{1}\n的结果：\n{2}'.format(m1, m2, result))

    # 3. 随机生成100个标准正态的数据，并计算数据的均值与标准差
    norm_data = np.random.randn(100)
    print('\n随机生成的100个标准正态的数据:\n{0}\n数据的均值:\t   {1}\n{2}标准差:\t{3}'.
          format(norm_data, norm_data.mean(), ' ' * 6, norm_data.std()))

# classes

# main entry
if __name__ == "__main__":
    main()
    