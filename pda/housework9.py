#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework9.py

    DESCRIPTION
        1. 若要反驳以下论断，请写出下列论断的零假设与备择假设
            （1） 百事可乐公司声称，其生产的罐装可乐的标准差为0.005磅
            （2） 某社会调查员说从某项调查得知中国的离婚率高达38.5%
            （3） 某学校招生宣传手册中写道，该学校的学生就业率高达99%。

        2. 某学生随机抽取了10包一样的糖并称量它们的包装的重量，判断这些糖的包装的平均重量是否为3.5g。
        其中，这10包糖的重量如下（单位：g）：
            3.2,3.3,3.0,3.7,3.5,4.0,3.2,4.1,2.9,3.3

        3. 计算Amtrak.xls 数据集的均值、标准差、方差、最大值、最小值、25%分位数、75%分位数、偏度、峰度

    MODIFIED  (MM/DD/YY)
        Na  12/19/2018

"""
__VERSION__ = "1.0.0.12192018"


# imports
import pandas as pd
from pandas import Series, DataFrame
import os, unicodedata
import numpy as np
import matplotlib.pyplot as plt

# configuration
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format
plt.rc('figure', figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week9_data')

# functions
def main():
    # 1. 若要反驳以下论断，请写出下列论断的零假设与备择假设
    # （1） 百事可乐公司声称，其生产的罐装可乐的标准差为0.005磅
    # （2） 某社会调查员说从某项调查得知中国的离婚率高达38.5%
    # （3） 某学校招生宣传手册中写道，该学校的学生就业率高达99%。
    print(u'若要反驳以下论断，则设定:')
    print(u'（1） 百事可乐公司声称，其生产的罐装可乐的标准差为0.005磅\n\t零假设:\t\t{}\n\t备择假设:\t{}\n'.
          format(u'百事可乐公司生产的罐装可乐的标准差等于0.005磅', u'百事可乐公司生产的罐装可乐的标准差大于0.005磅'))
    print(u'（2） 某社会调查员说从某项调查得知中国的离婚率高达38.5%\n\t零假设:\t\t{}\n\t备择假设:\t{}\n'.
          format(u'中国的离婚率等于38.5%', u'中国的离婚率小于38.5%'))
    print(u'（3） 某学校招生宣传手册中写道，该学校的学生就业率高达99%\n\t零假设:\t\t{}\n\t备择假设:\t{}\n'.
          format(u'该学校的学生就业率等于99%', u'该学校的学生就业率小于99%'))

    # 2. 某学生随机抽取了10包一样的糖并称量它们的包装的重量，判断这些糖的包装的平均重量是否为3.5g。
    # 其中，这10包糖的重量如下（单位：g）：
    #     3.2,3.3,3.0,3.7,3.5,4.0,3.2,4.1,2.9,3.3
    from scipy import stats as ss
    df = DataFrame({'data': [3.2,3.3,3.0,3.7,3.5,4.0,3.2,4.1,2.9,3.3]})

    # check distribution shape, if similar to normal distribution
    df.plot(kind='hist', title=u'数据集分布')
    df.plot(kind='kde', ax=plt.gca())
    plt.show()
    print(u'数据集来自于N次伯努利试验，分布类似正态分布 => \033[1;32m符合t分布，用t检验处理。\033[0m')

    hmean = 3.5
    print(u'问题：某学生随机抽取了10包一样的糖并称量它们的包装的重量，判断这些糖的包装的平均重量是否为3.5g。')
    print(u'解答：\n1. 设定原假设：这些糖的包装的平均重量等于3.5g\n\t 备择假设：这些糖的包装的平均重量不等于3.5g')
    print(u'2. 设定检验统计值: 这些糖的平均重量 = {}'.format(hmean))
    t_statistic, p_value = ss.ttest_1samp(a=df, popmean=hmean)
    print(u'3. 计算得出：statistic = {}，p-value = {}'.format(t_statistic, p_value))

    # # calculate t manually
    # ddata = df.data
    # dmean = ddata.mean()
    # # 计算标准误差： 样本标准差 / （n的开方）
    # se = ddata.std() / np.sqrt(ddata.size)
    # # # 用 ss.sem() 计算
    # # se2 = ss.sem(ddata)
    # # print(u'手动计算的标准误差 == 用ss.sem() 计算的标准误差？{}'.format(se == se2))
    # t_manual = (dmean - hmean) / se
    # print('t_statistic_manually_calculated:\t{:,.4f}'.format(t_manual))
    # print(u'将t值和自由度v=n-1代入 Statistical distributions and interpreting P values\n\t'
    #       u'http://link.zhihu.com/?target=https%3A//www.graphpad.com/quickcalcs/distMenu/\n'
    #       u'中可得双尾t检验的p值为0.5450。')
    # #
    # # 根据自由度n-1和α查找t临界值表，计算1-α=95% 的置信水平
    # #   https://www.cnblogs.com/emanlee/archive/2008/10/25/1319520.html
    # #  t_statistic > t临界值t_ci 就是拒绝域。
    # # ref: https://zhuanlan.zhihu.com/p/29284854
    # # ref: https://zhuanlan.zhihu.com/p/36727517
    # #
    # t_ci = 2.262
    # a, b = dmean - t_ci * se, dmean + t_ci * se
    # print(u'根据自由度n-1和α查找t临界值表得到t临界值：\t{}\n计算得到95%的置信区间为：\t\t\t\t\t[{}, {}]\n'.
    #       format(t_ci, a, b))
    # # 计算效应量
    # d = (dmean - hmean) / ddata.std()
    # print(u'效应量:\td = {:,.4f}'.format(d))
    # d_res = u'大' if abs(d) >= 0.8 else (u'中' if 0.2 < abs(d) <0.8 else u'小')
    # print(u"查效应量Cohen's d绝对值和效果显著含义的对应表，可知：\t差异{}".format(d_res))

    alpha, alphacode = 0.05, unicodedata.lookup('GREEK SMALL LETTER ALPHA')
    print(u'   取{} = {}'.format(alphacode, alpha))
    result = u'因为p-value<={}, 所以拒绝原假设，这些糖的包装的平均重量不等于3.5g。'.format(alphacode) \
        if p_value <= alpha \
        else u'因为p-value>{}, 所以不拒绝原假设，即这些糖的包装的平均重量等于3.5g。'.format(alphacode)
    print(u'4. 得出结论：\033[1;31m{}\033[0m'.format(result))

    # 3. 计算Amtrak.xls 数据集的均值、标准差、方差、最大值、最小值、25%分位数、75%分位数、偏度、峰度
    fxls = os.path.join(DATA_PATH, r'Amtrak.xls')
    data = pd.read_excel(fxls, sheet_name='Data', parse_dates=True, index_col=0).dropna(axis=1)
    data_rs = data['Ridership']
    print(u'\nAmtrak.xls 数据集的统计信息：\n\t均值:\t   {:{fmt}}\n\t标准差:    {:{fmt}}\n\t方差:\t   {:{fmt}}\n\t'
          u'最大值   : {:{fmt}}\n\t最小值   : {:{fmt}}\n\t25%分位数: {:{fmt}}\n\t75%分位数: {:{fmt}}\n\t'
          u'偏度:\t   {:{fmt}}\n\t峰度:\t   {:{fmt}}'
          .format(data_rs.mean(), data_rs.std(), data_rs.var(), data_rs.max(), data_rs.min(), data_rs.quantile(0.25),
                  data_rs.quantile(0.75), data_rs.skew(), data_rs.kurt(), fmt='11,.4f'))

# main entry
if __name__ == "__main__":
    main()
