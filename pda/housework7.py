#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework7.py

    DESCRIPTION
         对macrodata.csv数据集
            1. 画出realgdp列的直方图
            2. 画出realgdp列与realcons列的散点图，初步判断两个变量之间的关系
        对tips数据集
            3. 画出不同sex与day的交叉表的柱形图
            4. 画出size的饼图

    MODIFIED  (MM/DD/YY)
        Na  12/08/2018

"""
__VERSION__ = "1.0.0.12082018"


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# configuration
plt.rc('figure', figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(precision=4)
pd.options.display.float_format = '{:,.4f}'.format
pd.options.display.max_rows = 100

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week7_data')

# functions
def main():
    #  对macrodata.csv数据集
    #      1. 画出realgdp列的直方图
    #      2. 画出realgdp列与realcons列的散点图，初步判断两个变量之间的关系
    # read in data
    fmdata = os.path.join(DATA_PATH, r'macrodata.csv')
    realgdp, realcons = np.loadtxt(fmdata, delimiter=',', usecols=(2, 3), unpack=True, skiprows=1)

    plt.ion()
    # hist for realgdp
    fig, axes = plt.subplots(2, 1)
    axes[0].hist(realgdp, bins=50, label='bins=50')
    axes[0].set_title(u'realgdp列的直方图')
    axes[0].legend()

    # scatter of realgdp, realcons
    axes[1].scatter(realgdp, realcons)
    axes[1].set_title(u'realgdp列与realcons列的散点图\n二阶多项式拟合（红色直线）显示两者线性相关')
    axes[1].set_xlabel('realgdp'); axes[1].set_ylabel('realcons')
    # use 2-deg polynomial fitting to check the linear correlation
    pcoef = np.polyfit(realgdp, realcons, 2)
    p2 = np.poly1d(pcoef)
    axes[1].plot(realgdp, p2(realgdp), 'r')
    fig.subplots_adjust(hspace=0.5)

    #  对tips数据集
    #      3. 画出不同sex与day的交叉表的柱形图
    #      4. 画出size的饼图
    # read in data
    ftips = os.path.join(DATA_PATH, r'tips.csv')
    tdata = pd.read_csv(ftips)
    cddata = pd.crosstab(tdata['sex'], tdata['day'])

    # bar
    fig, axes = plt.subplots(2, 1)
    cddata.plot(kind='barh', ax=axes[0])
    axes[0].set_title(u'sex与day的交叉表的柱形图')

    # pie
    vc = tdata['size'].value_counts().sort_index()
    plt.pie(vc.values, explode=[0.1] * 6, labels=vc.index.values, startangle=67)
    axes[1].set_title(u'size的饼图')
    fig.subplots_adjust(hspace=0.5)

    # close all plots
    plt.pause(5)
    plt.close('all')

# main entry
if __name__ == "__main__":
    main()
    