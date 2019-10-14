#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework11.py

    DESCRIPTION
         1. data1 是40名癌症病人的一些生存资料，其中，X1表示生活行动能力评分（1~100），X2表示病人的年龄，
         X3表示由诊断到直入研究时间（月）；X4表示肿瘤类型，X5把ISO两种疗法（“1”是常规，“0”是试验新疗法）；
         Y表示病人生存时间（“0”表示生存时间小于200天，“1”表示生存时间大于或等于200天）
        试建立Y关于X1~X5的logistic回归模型

        2. data2 是关于重伤病人的一些基本资料。自变量X是病人的住院天数，因变量Y是病人出院后长期恢复的预后指数，
        指数数值越大表示预后结局越好。
        尝试对数据拟合合适的线性或非线性模型

    MODIFIED  (MM/DD/YY)
        Na  01/05/2019

"""
__VERSION__ = "1.0.0.01052019"

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, os.path, sys
import pprint as pp

# configuration
np.set_printoptions(precision=4, threshold=500)
np.random.seed(12345)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format
plt.rc('figure', figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week11_data')

# functions
def main():
    # 1. data1 是40名癌症病人的一些生存资料，试建立Y关于X1~X5的logistic回归模型
    # read data
    fdata1 = os.path.join(DATA_PATH, r'data1.txt')
    data1 = pd.read_csv(fdata1, delimiter='\t', index_col=0, encoding='mbcs')
    x = data1.iloc[:, :-1].values
    y = data1.iloc[:, -1].values
    # Logistic Regression: all features in
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.metrics import roc_auc_score, mean_squared_error
    lr = LR()
    lr.fit(x, y)
    lry_pred = lr.predict(x)
    lrscore, lrmse = lr.score(x, y), mean_squared_error(y, lry_pred)
    print(u"题目1：\n\tY关于X1~X5逻辑回归模型结果：score={:,.4f}, MSE={:,.4f}".format(lrscore, lrmse))
    # draw Confusion Matrix to visualize the LR result
    def cm_plot(y, yp):
        from sklearn.metrics import confusion_matrix
        # import matplotlib.pyplot as plt
        cm = confusion_matrix(y, yp)
        plt.matshow(cm, cmap=plt.cm.Set2)
        plt.colorbar()
        cmlen = len(cm)
        for x in range(cmlen):
            for y in range(cmlen):
                plt.annotate(cm[x, y], xy=(x, y))
        plt.title(u'Y与逻辑回归预测结果Y_pred混淆矩阵图')
        plt.xlabel(u'预测值'); plt.ylabel(u'实际值')
        return plt
    plt.ion()
    cm_plot(y, lry_pred).pause(5)


    # 2. data2 是关于重伤病人的一些基本资料。尝试对数据拟合合适的线性或非线性模型
    # read data
    fdata2 = os.path.join(DATA_PATH, r'data2.txt')
    data2 = pd.read_csv(fdata2, delimiter='\t', index_col=0, encoding='mbcs')
    d2x, d2y = pd.DataFrame(data2.X), pd.DataFrame(data2.Y)
    # Linear Regression
    from sklearn.linear_model import LinearRegression as LiR
    linreg = LiR()
    linreg.fit(d2x, d2y)
    lind2y_pred = linreg.predict(d2x)
    linscore, linsme = linreg.score(d2x, d2y), mean_squared_error(d2y, lind2y_pred)
    msg = u'Linear Regression：score={:,.4f}, MSE={:,.4f}'.format(linscore, linsme)
    print(u'题目2:\n\t{}'.format(msg))
    fig = plt.figure(2)
    plt.scatter(d2x, d2y, label=u'original data')
    plt.plot(d2x, lind2y_pred, label=msg)

    # Logistic Regression
    lr2 = LR()
    lr2.fit(d2x, d2y)
    lrd2y_pred = lr2.predict(d2x)
    lrd2score, lrd2sme = lr2.score(d2x, d2y), mean_squared_error(d2y, lrd2y_pred)
    msg = u'Logistic Regression：score={:,.4f}, MSE={:,.4f}'.format(lrd2score, lrd2sme)
    print(u'\t{}'.format(msg))
    plt.plot(d2x, lrd2y_pred, label=msg)

    # Polynomial Regression
    d2xc = d2x.copy()
    d2xc['X2'] = d2x**2
    plinreg = LiR()
    plinreg.fit(d2xc, d2y)
    plind2y_pred = plinreg.predict(d2xc)
    plinscore, plinsme = plinreg.score(d2xc, d2y), mean_squared_error(d2y, plind2y_pred)
    msg = u'Polynomial Regression：score={:,.4f}, MSE={:,.4f}'.format(plinscore, plinsme)
    print(u'\t{}'.format(msg))
    plt.plot(d2x, plind2y_pred, label=msg)

    # Log Regression
    logx = np.log(d2x)
    logreg = LiR()
    logreg.fit(logx, d2y)
    logd2y_pred = logreg.predict(logx)
    logscore, logsme = logreg.score(logx, d2y), mean_squared_error(d2y, logd2y_pred)
    msg = u'Log Regression：score={:,.4f}, MSE={:,.4f}'.format(logscore, logsme)
    print(u'\t{}'.format(msg))
    plt.plot(d2x, logd2y_pred, label=msg)

    # Exponent Regression
    logy = np.log(d2y)
    expreg = LiR()
    expreg.fit(d2x, logy)
    expd2y_pred = expreg.predict(d2x)
    expscore, expsme = expreg.score(d2x, logy), mean_squared_error(logy, expd2y_pred)
    msg = u'Exponent Regression：score={:,.4f}, MSE={:,.4f}'.format(expscore, expsme)
    print(u'\t{}'.format(msg))
    plt.plot(d2x, np.exp(expd2y_pred), label=msg)

    # Power Regression
    powreg = LiR()
    powreg.fit(logx, logy)
    powd2y_pred = powreg.predict(logx)
    powscore, powsme = powreg.score(logx, logy), mean_squared_error(logy, powd2y_pred)
    msg = u'Power Regression：score={:,.4f}, MSE={:,.4f}'.format(powscore, powsme)
    print(u'\t{}'.format(msg))
    plt.plot(d2x, np.exp(powd2y_pred), label=msg)

    plt.legend()
    plt.title(u'题目2：对重伤病人的数据的线性或非线性模型拟合图示')
    print(u'从score 和 MSE 大小 和画图展示得知：\n\t\033[1;31mPolynomial Regression 和 Exponent Regression 拟合结果比较好\033[0m。')
    plt.pause(5)
    plt.close('All')


# main entry
if __name__ == "__main__":
    main()
    