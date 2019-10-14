#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework8.py

    DESCRIPTION
        读入tips.csv 数据集
        1. 统计不同time的tip的均值，方差
        2. 将total_bill和tip根据不同的sex进行标准化(原数据减去均值的结果除以标准差)
        3. 计算吸烟者和非吸烟者的小费比例值均值  的差值
        4. 对sex与size聚合，统计不同分组的小费比例的标准差、均值，将该标准差与均值添加到原数据中
        5. 对time和size聚合，画出total_bill 的饼图

    MODIFIED  (MM/DD/YY)
        Na  12/16/2018

"""
__VERSION__ = "1.0.0.12162018"


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pprint as pp

# configuration
np.random.seed(12345)
np.set_printoptions(precision=4, threshold=500)
pd.options.display.max_rows=100
pd.options.display.float_format='{:,.4f}'.format
plt.rc('figure', figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# consts
CURRENT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(CURRENT_PATH, r'data', r'week8_data')

# functions
def main():
    # 读入tips.csv 数据集
    ftips = os.path.join(DATA_PATH, r'tips.csv')
    data = pd.read_csv(ftips)

    # 1. 统计不同time的tip的均值，方差
    by_time = data.groupby('time')
    print(u"不同time的tip的均值，方差:\n{}".format(by_time.tip.agg(['mean', 'var'])))

    # 2. 将total_bill和tip根据不同的sex进行标准化(原数据减去均值的结果除以标准差)
    by_sex = data.groupby('sex')
    def to_norm(group):
        return (group - group.mean()) / group.std()
    grouped = by_sex[['total_bill', 'tip']]
    print(u"\ntotal_bill和tip根据不同的sex进行标准化(原数据减去均值的结果除以标准差)以后:\n{}".
          format(grouped.apply(to_norm)))

    # 3. 计算吸烟者和非吸烟者的小费比例值均值  的差值
    data_with_tp = data.assign(tip_percent = data.tip / data.total_bill)
    tp_smoker_mean = data_with_tp.groupby('smoker').tip_percent.mean()
    print(u"\n吸烟者和非吸烟者的小费比例值均值的差值:\n{:,.4f}".format(tp_smoker_mean.Yes - tp_smoker_mean.No))

    # 4. 对sex与size聚合，统计不同分组的小费比例的标准差、均值，将该标准差与均值添加到原数据中
    by_ss = data_with_tp.groupby(['sex', 'size'])
    agg_std_mean = by_ss.tip_percent.agg(['std', 'mean'])
    data['sex_size_tippct_std'] = by_ss.tip_percent.transform('std')
    data['sex_size_tippct_mean'] = by_ss.tip_percent.transform('mean')

    # 5. 对time和size聚合，画出total_bill 的饼图
    tb_by_ts = data.groupby(['time', 'size']).total_bill
    # plot
    plt.ion()
    values = tb_by_ts.mean()
    labels = ['{} in size {}\navg ${:,.2f}'.format(k[0], k[1], v) for k, v in dict(values).iteritems()]
    values.plot(kind='pie', labels=labels,autopct='%1.1f%%', explode=[0.05] * len(tb_by_ts), startangle=67,
                title=u'对time和size聚合以后total_bill平均数额的饼图')
    # close all plots
    plt.pause(5)
    plt.close('all')

# main entry
if __name__ == "__main__":
    main()
    