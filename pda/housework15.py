#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework15.py

    DESCRIPTION
         ex15.txt 记录了进口总额Y与三个自变量：国内生产总值X1、存储量X2、总消费X3 的值。
        先对自变量进行主成分法分析，然后将得到的主成分与Y进行回归分析
        试比较与直接进行回归分析所得到的结果差异，说说哪个模型的拟合效果比较好

    MODIFIED  (MM/DD/YY)
        Na  01/29/2019

"""
__VERSION__ = "1.0.0.01292019"


# imports
import os
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# configuration
plt.rc('figure', figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week15_data')

# functions
def main():
    # ex15.txt 记录了进口总额Y与三个自变量：国内生产总值X1、存储量X2、总消费X3 的值。
    # read in data
    fex15 = os.path.join(DATA_PATH, r'ex15.txt')
    data = pd.read_csv(fex15, delimiter='\s+', index_col=0)
    X, y = data.iloc[:, :3], data.iloc[:, 3]

    # 对自变量进行主成分法分析，然后将得到的主成分与Y进行回归分析
    k = 2
    pca = PCA(n_components=k)
    reduced_X = pca.fit_transform(X)
    reduced_lr = LinearRegression()
    reduced_lr.fit(reduced_X, y)
    reduced_predict_y = reduced_lr.predict(reduced_X)
    lr = LinearRegression()
    lr.fit(X, y)
    predict_y = lr.predict(X)

    # 试比较与直接进行回归分析所得到的结果差异，说说哪个模型的拟合效果比较好
    reduced_score = reduced_lr.score(reduced_X, y)
    reduced_mse = mean_squared_error(y, reduced_predict_y)
    score = lr.score(X, y)
    mse = mean_squared_error(y, predict_y)
    print(u'对自变量进行主成分法分析n_components={}得到的主成分与Y进行回归分析：'
          u'\n\tscore:\t{:,.4f}\n\tMSE:\t{:,.4f}'.format(k, reduced_score, reduced_mse))
    print(u'直接回归分析：\n\tscore:\t{:,.4f}\n\tMSE:\t{:,.4f}'.format(score, mse))

    # draw the LR fit linear
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    ax.plot(y, y, color='r', label=u'原数据')
    ax.plot(y, reduced_predict_y, color='b',
            label=u'对自变量进行主成分法分析n_components={}得到的主成分与Y进行回归分析'.format(k))
    ax.plot(y, predict_y, color='g', label=u'直接回归分析')
    fig.suptitle(u'利用主成分法分析先降维再回归分析 和 直接回归分析的对比')
    ax.legend()
    plt.pause(5)

    print(u'从两种模型的结果（score、MeanSquaredError)和图示可以看出：'
          u'\n\t\033[1;31m直接回归分析的拟合效果比较好。\033[0m'
          u'\n\t<= 原因可能为：1. 样本数量太少（{}个标本{}个自变量）'
          u'\n\t              2. 主成分分析降维损失了有用信息'
          u'\n\t=> 样本数量太少、或者主成分分析损失有用信息，最好不使用降维方法。'.format(*data.shape))

    # close the plots
    plt.close()


# main entry
if __name__ == "__main__":
    main()
    