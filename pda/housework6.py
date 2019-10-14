#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework6.py

    DESCRIPTION
        1. 读入  肝气郁结证型系数.xls  数据集，将数据集按照等距、小组等量 两种方式 分别分为5组数据，
        分别计算5组数据的中位数与标准差
        2. 读入BHP1.csv，使用适当的方法填补缺失值
        3. 读入BHP2.xlsx，与BHP1数据集合并为BHP数据集
        4. 将BHP数据集中的成交量（volume）替换为 high、median、low 三种水平（区间自行定义）

    MODIFIED  (MM/DD/YY)
        Na  12/02/2018

"""
__VERSION__ = "1.0.0.12022018"

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, os.path
import pprint as pp

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, u'data\第6周作业数据')

# functions
def main():
    # 1. 读入  肝气郁结证型系数.xls  数据集，将数据集按照等距、小组等量 两种方式 分别分为5组数据，
    # 分别计算5组数据的中位数与标准差
    f_liver = os.path.join(DATA_PATH, u'肝气郁结证型系数.xls')
    data = pd.read_excel(f_liver)
    print(u'读入{}:\n{}'.format(f_liver, data))
    N, dv = 5, data.values.ravel()
    data_edist = pd.cut(dv, N)
    print('数据集按照等距分为5组数据:\n{}\n'.format(data_edist))
    for i in xrange(N):
        cdv = dv[data_edist.codes == i]
        print('\t{}中位数:\t{}\t标准差:\t{}'.format(data_edist[i], np.median(cdv), np.std(cdv)))
    data_ecount = pd.qcut(dv, N)
    print('数据集按照小组等量分为5组数据:\n{}\n'.format(data_ecount))
    for i in xrange(N):
        cdv = dv[data_ecount.codes == i]
        print('\t{}中位数:\t{}\t标准差:\t{}'.format(data_ecount[i], np.median(cdv), np.std(cdv)))

    # 2. 读入BHP1.csv，使用适当的方法填补缺失值
    f_bhp1 = os.path.join(DATA_PATH, u'BHP1.csv')
    d_bhp1 = pd.read_csv(f_bhp1,squeeze=True)
    print(u'读入{}:\n{}'.format(f_bhp1, d_bhp1))
    d_bhp1.rename(columns={u'Unnamed: 7': u'volume'}, inplace=True)
    d_bhp1[u'volume'] = 0
    d_bhp1.fillna(method='ffill', inplace=True)
    print(u'把最后一列命名为volume，全设为0，并用前一天的数值填补缺失值以后:\n{}'.format(d_bhp1))

    # 3. 读入BHP2.xlsx，与BHP1数据集合并为BHP数据集
    f_bhp2 = os.path.join(DATA_PATH, u'BHP2.xlsx')
    d_bhp2 = pd.read_excel(f_bhp2)
    print(u'读入{}:\n{}'.format(f_bhp2, d_bhp2))
    all_bhp = pd.concat([d_bhp1, d_bhp2], axis=0, ignore_index=True, sort=False)
    print(u'与BHP1数据集合并为BHP数据集:\n{}'.format(all_bhp))

    # 4. 将BHP数据集中的成交量（volume）替换为 high、median、low 三种水平（区间自行定义）
    volume = all_bhp[u'volume']
    print(u'BHP数据集中的成交量（volume）按照原数据集中的数字形式:\n{}'.format(volume))
    vol_0 = np.where(volume == 0)[0]
    # divide volume into 3 levels 'low', 'median', 'high' via pd.qcut using volume > 0
    v_qcut = pd.qcut(volume.drop(vol_0), 3, labels=['low', 'median', 'high'])
    # set volume =0 into 'low' level
    for i in vol_0:
        v_qcut.ix[i] = 'low'
    # update volume in BHP dataset to 3 levels
    all_bhp[u'volume_orig'] = all_bhp[u'volume']
    all_bhp[u'volume'] = v_qcut.sort_index()
    print(u'BHP数据集中的成交量（volume）替换为 high、median、low 三种水平以后（区间按照小组等量划分）:\n{}'.
          format(all_bhp))

# classes

# main entry
if __name__ == "__main__":
    main()
    