#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework10.py

    DESCRIPTION
        1. 基于最大积雪深度 X 与当年灌溉面积 Y 的数据，
            （1）画出x与y的散点图，初步判断x与y的关系
            （2）求出Y关于X的一元线性方程
            （3）若今年的X=7，估计Y的值
        2. 基于某地区土壤所含可给态磷的情况，
            求出关于Y的多元线性模型，并尝试删除某一变量后，与全变量的线性回归方程进行比较，找出最优模型

    MODIFIED  (MM/DD/YY)
        Na  12/27/2018

"""
__VERSION__ = "1.0.0.12272018"

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# configuration
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, threshold=500)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# functions
def main():
    # 1. 基于最大积雪深度 X 与当年灌溉面积 Y 的数据，
    X = np.array([5.1, 3.5, 7.1, 6.2, 8.8, 7.8, 4.5, 5.6, 8.0, 6.4])
    Y = np.array([1907, 1287, 2700, 2373, 3260, 3000, 1947, 2273, 3113, 2493])

    #  （1）画出x与y的散点图，初步判断x与y的关系
    # corrcoef
    corrcoef = np.corrcoef(X, Y)
    print(u'题目1：\nX与Y的相关系数:\t{:,.4f}'.format(corrcoef[0][1]))
    plt.ion()
    plt.scatter(X, Y)
    plt.title(u'X与Y的散点图\n(相关系数: {:,.4f})'.format(corrcoef[0][1])); plt.xlabel('X'); plt.ylabel('Y')
    plt.pause(5)
    plt.close('all')

    #  （2）求出Y关于X的一元线性方程
    # calculate manually
    b = np.sum((X - X.mean()) * (Y - Y.mean())) / np.sum((X - X.mean()) ** 2)
    a = Y.mean() - b * X.mean()
    print(u'手动计算的Y关于X的一元线性方程：\t\tY = {:,.4f} * X + {:,.4f}'.format(b, a))
    # calculate using sklearn
    # use all X, Y to fit OLS
    linreg = LinearRegression()
    linreg.fit(X.reshape(-1, 1), Y)
    aa, bb = linreg.coef_[0], linreg.intercept_
    print(u'用sklearn计算的Y关于X的一元线性方程：Y = {:,.4f} * X + {:,.4f}'.format(aa, bb))

    #  （3）若今年的X=7，估计Y的值
    print(u'若今年的 \033[1;31mX=7\033[0m，可以估计Y的值为： \033[1;31m{:,.0f}\033[0m\n\n'.format(aa * 7 + bb))

    # divide X, Y to train, test set and then fit OLS and do model-evaluation
    #   <= when dataset is small, not as precise as to use all X, Y to fit OLS.
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X, Y, random_state=1)
    linreg2 = LinearRegression()
    linreg2.fit(Xs_train.reshape(-1, 1), ys_train)
    aa2, bb2 = linreg2.coef_[0], linreg2.intercept_
    s_score = linreg2.score(Xs_test.reshape(-1, 1), ys_test)
    print(u'用sklearn划分训练集和测试集以后计算的Y关于X的一元线性方程：Y = {:,.4f} * X + {:,.4f}'.format(aa2, bb2))
    # print(u'用sklearn划分训练集和测试集以后预测值：若今年的 \033[1;31mX=7\033[0m，可以估计Y的值为： '
    #       u'\033[1;31m{:,.0f}\033[0m\n\n'.format(aa2 * 7 + bb2))
    print(u'用sklearn划分训练集和测试集以后预测值：若今年的 \033[1;31mX=7\033[0m，可以估计Y的值为： '
          u'\033[1;31m{:,.0f}\033[0m\n\n'.format(linreg2.predict([[7]])[0]))

    # 2. 基于某地区土壤所含可给态磷的情况，
    pdata = pd.DataFrame(
        data={'X1': [0.4, 0.4, 3.1, 0.6, 4.7, 1.7, 9.4, 10.1, 11.6, 12.6, 10.9, 23.1, 23.1, 21.6, 23.1, 1.9, 26.8, 29.9],
              'X2': [52, 23, 19, 34, 24, 65, 44, 31, 29, 58, 37, 46, 50, 44, 56, 36, 58, 51],
              'X3': [158, 163, 37, 157, 59, 123, 46, 117, 173, 112, 111, 114, 134, 73, 168, 143, 202, 124],
              'Y': [64, 60, 71, 61, 54, 77, 81, 93, 93, 51, 76, 96, 77, 93, 95, 54, 168, 99]},
    index=np.arange(1, 19))

    # # 对多元线性模型，可以用 sns.pairplot(...) 画出散点图，并计算相关系数作为参考
    # print(u'Y和各X变量的相关系数：\n{}\n'.format(pdata.corr()))
    # import seaborn as sns
    # sns.pairplot(pdata, x_vars=features, y_vars=['Y'], size=7, aspect=0.8, kind='reg')

    # 求出关于Y的多元线性模型，
    # all features in
    features = ('X1', 'X2', 'X3')
    Xf = pdata[list(features)]
    y = pdata.Y
    # calculate with all features
    Xf_train, Xf_test, y_train, y_test = train_test_split(Xf, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(Xf_train, y_train)
    # error evaluation
    y_pred = linreg.predict(Xf_test)
    rmsef = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    coeff = zip(features, linreg.coef_)
    score = linreg.score(Xf_test, y_test)
    print(u'题目2：\nY与全变量{}的多元线性模型：\tY = {} + {:,.4f}，RMSE = {:,.4f}，模型评分 = {:,.4f}'.format(
        features, ' + '.join([' * '.join(('{:,.4f}'.format(b), a)) for a, b in coeff]), linreg.intercept_, rmsef,
        score))

    # 并尝试删除某一变量后，与全变量的线性回归方程进行比较，找出最优模型
    # track the optimal parameters
    best_coef, best_intercept = linreg.coef_, linreg.intercept_
    best_features, best_rmse = features, rmsef
    best_score = score
    # draw the ols-scatter
    fig, axes = plt.subplots(1, len(features) + 1)
    fig.suptitle(u'Y与变量$X_n$的多元线性模型拟合曲线散点图')
    axes[0].scatter(y_test.values, y_pred)
    axes[0].set_title(u'Y与{}'.format(features))
    idx = 1

    from itertools import combinations

    # calculate with less features
    for xab in combinations(features, len(features) - 1):
        # calculate regression coef_ and intercept_
        Xab = pdata[list(xab)]
        Xab_train, Xab_test, yab_train, yab_test = train_test_split(Xab, y, random_state=1)
        linreg.fit(Xab_train, yab_train)
        yab_pred = linreg.predict(Xab_test)
        rmseab = np.sqrt(metrics.mean_squared_error(yab_test, yab_pred))
        coeff, scoreab = zip(xab, linreg.coef_), linreg.score(Xab_test, yab_test)
        print(u'Y与{}的多元线性模型：\t\t\t\tY = {} + {:,.4f}，RMSE = {:,.4f}，模型评分 = {:,.4f}'.format(
            xab, ' + '.join([' * '.join(('{:,.4f}'.format(b), a)) for a, b in coeff]), linreg.intercept_, rmseab,
            scoreab))
        # compare to get the optimal model
        if abs(rmseab) < abs(best_rmse):
            best_coef, best_intercept = linreg.coef_, linreg.intercept_
            best_features, best_rmse, best_score = xab, rmseab, scoreab
        else:
            pass
        # draw the ols-scatter
        axes[idx].scatter(yab_test, yab_pred)
        axes[idx].set_title(u'Y与{}'.format(xab))
        idx += 1

    # output the optimal model
    coeff = zip(best_features, best_coef)
    print(u'\n=> 从3个角度：拟合曲线的散点图、RMSE最小、模型评分最高可以得出结论：')
    print(u'\t最优模型为Y与{}的多元线性模型：\t\033[1;31mY = {} + {:,.4f}\033[0m，'
          u'其 RMSE 最小 = {:,.4f}，模型评分 = {:,.4f}'.format(
        best_features, ' + '.join([' * '.join(('{:,.4f}'.format(b), a)) for a, b in coeff]),
        best_intercept, best_rmse, best_score))

    # close plots
    plt.pause(5)
    plt.close('all')


# main entry
if __name__ == "__main__":
    main()
    