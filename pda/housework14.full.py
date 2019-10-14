#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework14.py

    DESCRIPTION
        数据集 ex14.csv 是关于中国各个省份的三项指标数值。
        请根据这些指标数值，将各个省份分为3类，并尝试归纳出各个类别的特点

    MODIFIED  (MM/DD/YY)
        Na  01/27/2019

"""
__VERSION__ = "1.0.0.01272019"


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
# 层次聚类函数
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# configuration
np.set_printoptions(precision=4, suppress=True, threshold=100)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format
plt.rc('figure', figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week14_data')
# CURRENT_PATH, DATA PATH here is for running code snippet in Python Console by ^+Alt+E
CURRENT_PATH = os.getcwd()
DATA_PATH = os.path.join(CURRENT_PATH, r'pda\data\week14_data')

# functions
def main():
    # 数据集 ex14.csv 是关于中国各个省份的三项指标数值。
    fex14 = os.path.join(DATA_PATH, r'ex14.csv')
    data = pd.read_csv(fex14, index_col=0, encoding='mbcs')
    # 离差标准化
    data = (data - data.min()) / (data.max() - data.min())

    # 请根据这些指标数值，将各个省份分为3类
    plt.ion()
    # 谱系聚类图
    Z = linkage(data, method='ward', metric='euclidean')
    # 画谱系聚类图
    P = dendrogram(Z, 0)
    plt.title(u'省份的谱系聚类图')

    # 聚类数
    k = 3
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    model.fit(data)
    # 详细的原始数据及其类别
    r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    r.columns = list(data.columns) + [u'省份类别']

    # 对每一类：逐一作图，采用不同样式
    style = ['ro-', 'go-', 'bo-']
    xlabels = list(data.columns)
    for i in range(k):
        plt.figure()
        tmp = r[r[u'省份类别'] == i].iloc[:, :3]
        for j in range(len(tmp)):
            plt.plot(range(1, 4), tmp.iloc[j], style[i])
        plt.xticks(range(1, data.columns.size+1), xlabels, rotation=20)
        plt.title(u'省份类别{}'.format(i+1))
        plt.subplots_adjust(bottom=0.15)
        # 保存成图片
        fsave = os.path.join(DATA_PATH, u'province_type_{}.png'.format(i+1))
        plt.savefig(fsave)

    # 并尝试归纳出各个类别的特点
    # 计算各个类别的数据特征
    grp = r.groupby(u'省份类别')
    gmean, gmin, gmax, gstd, gmedian = grp.mean(), grp.min(), grp.max(), grp.std(), grp.median()
    cols = data.columns
    for i in xrange(k):
        datalistall = []
        for c in cols:
            datalist1 = []
            datalist1.extend((u'平均值: {:,.4f}'.format(gmean[c][i]),
                                u'标准差: {:,.4f}'.format(gstd[c][i]),
                                u'最小值: {:,.4f}'.format(gmin[c][i]),
                                u'最大值: {:,.4f}'.format(gmax[c][i]),
                                u'中位数: {:,.4f}'.format(gmedian[c][i])))
            datalistall.append(u'{}:\n\t\t{}'.format(c, '\n\t\t'.join(datalist1)))
        clsinfostr = u'省份类别{}:\n\t{}个省份: {}\n\t{}'.format(
            i+1,
            grp.groups[i].size,
            ','.join(grp.groups[i]),
            '\n\t'.join(datalistall)
        )
        print(clsinfostr)
	# 归纳出各个类别的特点
	summarystr = u'处于类别2的省份最多，20个；其次是类别1，7个，类别3最少，3个且都是直辖市；' \
				 u'\n\t从DXBZ角度看:\n\t\t类别1的平均值在3个类别中最小，也在最低的范围内波动；' \
				 u'\n\t\t其次是类别2，平均值约是类别1的1.8倍；' \
				 u'\n\t\t类别3的平均值比类别1和类别2高一个数量级左右，分别是类别1的7.6倍、类别2的13.5倍；' \
				 u'\n\t从CZBZ角度看:\n\t\t类别1的平均值在3个类别中最小，也在最低的范围内波动；' \
				 u'\n\t\t其次是类别2，平均值约是类别1的1.8倍；' \
				 u'\n\t\t类别3的平均值是类别2的1.3倍、类别1的2.3倍；' \
				 u'\n\t从WMBZ角度看:\n\t\t类别3的平均值在3个类别中最小，也在最低的范围内波动，比类别1和类别2低一个数量级左右；' \
				 u'\n\t\t其次是类别2，平均值约是类别3的6.3倍；' \
				 u'\n\t\t类别1的平均值是类别2的3.6倍、类别3的22.6倍。'
	print(u'\n=> {}'.format(summarystr))

    plt.pause(5)
    plt.close()

# main entry
if __name__ == "__main__":
    main()
    