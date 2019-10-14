#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        reganalysis_practice.py

    DESCRIPTION
        1. 回归分析regression analysis： 通过建立模型来研究变量之间相互关系的密切程度、结构状态及进行模型预测的一种有效工具。
            可分6部分讨论：
            1) 线性回归
                一元线性回归
                多元线性回归
                    Y = βX + ε, β= [β₀, ..., βn], X= [X₀, ..., Xn].T,
                    也可以用最小二乘法进行参数估计，与一元回归方程的算法相似：
                        beta_head = (X_T * X).I * X_T * y
                    多元线性回归的参数估计也可以用梯度下降法估计，最小二乘法计算全局最优解，梯度下降法是局部最优解。
                        梯度下降法的一个重要超参数是步长（learning rate），用来控制蒙眼人步子的大小，就是下降幅度。
                        如果按照每次迭代后用于更新模型参数的训练样本数量划分，有两种梯度下降法。
                            批量梯度下降 Batch gradient descent （BGD）每次迭代都用所有训练样本。
                            随机梯度下降 Stochastic gradientdescent （SGD）每次迭代都用一个训练样本，
                                这个训练样本是随机选择的。
                            当训练样本较多的时候，随机梯度下降法比批量梯度下降法更快找到最优参数。
                            批量梯度下降法一个训练集只能产生一个结果。 而 SGD 每次运行都会产生不同的结果。
                                SGD 也可能找不到最小值，因为升级权重的时候只用一个训练样本。
                                它的近似值通常足够接近最小值，尤其是处理残差平方和这类凸函数的时候。


                多个因变量与多个自变量的回归
                    函数关系是确定性关系，相关关系是非确定性关系。
                    用相关系数衡量线性相关性的强弱。如果X与Y之间存在较强的相关关系，则有 Y≈α+βX+ε,
                        其中，截距项α， 斜率β， 误差项ε。
                    使用误差平方和 RSS 衡量预测值与真实值的差距。<=> 寻找合适的参数，使得误差平方和 RSS 最小。
                        RSS = Σᵢⁿ₌ ₁ (yᵢ - ŷᵢ)²， 其中真实值y，预测值ŷ。
                        使用最小二乘法表示 RSS = ...，即把 Y≈α+βX 代入 ŷᵢ，
                            即 RSS 其实是关于α与β的函数，分布对α与β求偏导并令偏导等于0，就可以得出α与β的值。
                            由于总体未知，采用样本值估计：
                                b = beta_head = sum((xi - x_bar) * (yi - y_bar)) / sum((xi - x_bar) ** 2)
                                                即：beta_head = cov(x, y) / var(x) = Sxy / var(x)
                                如果用矩阵计算， 即 beta_head = (X_T * X).I * X_T * y。

                                a = alpha_head = y_bar - b * x_bar
                        从而，对于每个xi, 可以通过 y_head_i = a + b * xi 预测相应的 y 值。

                    通常使用误差平方和 RSS 为成本函数。

            2) 回归诊断


            3) 回归变量选择
                自变量选择的标准: R^2 (R-square)等，AIC准则、BIC准则等等
                逐步回归分析法
                模型的残差满足3条件：独立、正态分布、均值为0，方差固定值。

            4) 参数估计方法改进
                    一元线性回归拟合模型的参数估计常用方法是普通最小二乘法OLS（ordinary least squares ）
                                                          或线性最小二乘法LLS（linear least squares）
                # ref:  https://www.jianshu.com/p/738f6092ef53

                当自变量之间高度相关的时候，最小二乘回归估计的参数估计值会不稳定；
                偏最小二乘回归：如果 自变量之间高度相关 + 例数特别少 + 自变量又很多，可以用偏最小二乘回归。
                        偏最小二乘回归比 主成分回归 和 岭回归RR 更好的一个优点是：可以用于例数很少的情形。
                        偏最小二乘回归的原理与 主成分回归 有点像，即提取自变量的部分信息，损失一定的精度，
                            但保证模型更符合实际。
                        偏最小二乘回归可用于多个因变量和多个自变量的分析，因为它同时提取多个因变量和多个自变量
                            重新组成新的变量重新分析。

                岭回归 Ridge Regression(RR)： 是最小二乘的改进；是一种有偏参数估计；
                         岭回归通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价获得更为符合实际、更可靠的回归系数。
                         岭回归增加 L2 范数项（相关系数向量平方和）来调整成本函数（残差平方和）。

                      最小收缩和选择算子即LASSO 算子 = Least Absolute Shrinkage and Selection Operator,
                        增加 L1 范数项（相关系数向量平方和的平方根）来调整成本函数（残差平方和）。

                      LASSO 方法会产生稀疏参数，大多数相关系数会变成 0，模型只会保留一小部分特征。
                      而岭回归还是会保留大多数尽可能小的相关系数。
                      当两个变量相关时，LASSO 方法会让其中一个变量的相关系数会变成 0，而岭回归是将两个系数同时缩小。
                      scikit-learn 还提供了弹性网（elastic net）正则化方法，通过线性组合 L1 和 L2 兼具 LASSO 和岭回归的内容。
                        可以认为这两种方法是弹性网正则化的特例。

                    正则化（Regularization）是用来防止拟合过度的一堆方法。 正则化向模型中增加信息，经常是一种对抗复杂性的手段。
                        与奥卡姆剃刀原理（Occam's razor）所说的具有最少假设的论点是最好的观点类似。
                        正则化就是用最简单的模型解释数据。岭回归RR、LASSO算子都是正则化的方法之一。

                    作者：wyrover
                    链接：https://www.jianshu.com/p/738f6092ef53
                    來源：简书
                    简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。

                主成分回归


            5) 非线性回归
                一元非线性回归
                    在回归分析中，只包括一个自变量和一个因变量，且二者的关系可用一条曲线近似表示，则称为一元非线性回归分析。
                    一元非线性回归可以通过多项式回归计算：
                            from sklearn.preprocessing import PolynomialFeatures
                            以求解 y=ax²+bx+c(二次方) 为例：
                            构建自变量的2阶数据：p2 = PolynomialFeatures(degree=2);
                                                p2.fit_transform(x_train.reshape(-1, 1))..., p2.transform(x_test...);
                    # ref: https://zhuanlan.zhihu.com/p/32994636

                分段回归
                多元非线性回归


            6) 含有定性变量的线性回归
                譬如用哑变量处理定性变量；Logistics 模型中，因变量或者分类变量可能是定性变量。

            相应的，可以建立线性模型和非线性模型，可以通过最小二乘法计算参数。
            Logistics 回归是广义线性回归模型的一种。

        2. Python 实现
            6个步骤：
               1) 画出散点图，计算相关系数： corrcoef = np.corrcoef(X, Y)
                    对多元线性模型，可以用 sns.pairplot(...) 画出散点图，并计算相关系数作为参考
                        print(u'Y和各X变量的相关系数：\n{}\n'.format(pdata.corr()))
                        # # [only for information] plot the model
                        # # [too many plots] pd.scatter_matrix(pdata, diagonal='kde', color='k', alpha=0.3)
                        # import seaborn as sns
                        # # [NOT needed] sns.pairplot(pdata, x_vars=features, y_vars=['Y'], size=7, aspect=0.8)
                        # sns.pairplot(pdata, x_vars=features, y_vars=['Y'], size=7, aspect=0.8, kind='reg')

                2) 选择变量，如用 sklearn.linear_model.RandomizedLogisticRegression.get_support() 选择一下：
                        X = data[['TV', 'Radio', 'Newspaper']], X = data[['TV', 'Radio']]
                3) 构建训练集与测试集: sklearn.model_selection.train_test_split(X, y, random_state=1)
                4) 训练数据 .fit(...train...)
                5）评估模型： 3个角度： 最小二乘拟合曲线的散点图、RMSE最小、模型评分最高 .score(...test...)
                    # RMSE最小
                    MAE = MeanAbsoluteError,
                    MSE = MeanSquaredError
                    RMSE = np.sqrt(MeanSquaredError)

                    rmsef = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                    coeff = zip(features, linreg.coef_)
                    # 模型评分最高
                    #   模型评分 .score(...), 即决定系数R平方：
                    #           有多少百分比的 y 波动没有被回归线描述 = 误差平方和/总波动
                    #           有多少百分比的 y 波动被回归线描述 = 1 - 误差平方和/总波动 ≈ 决定系数R平方
                    score = linreg.score(Xf_test, y_test)
                    # https://zhuanlan.zhihu.com/p/38178150

                    # 最小二乘拟合曲线的散点图
                    # draw the ols-scatter
                    fig, axes = plt.subplots(1, len(features) + 1)
                    fig.suptitle(u'Y与变量$X_n$的多元线性模型拟合曲线散点图')
                    # [scatter is more obvious than plot]
                    # ols_line = zip(y_test.values, y_pred)
                    # ols_line.sort()
                    # axes[0].plot([x[0] for x in ols_line], [x[1] for x in ols_line])
                    # axes[0].set_title(u'Y与全变量{}的多元线性模型最小二乘拟合曲线'.format(features))
                    axes[0].scatter(y_test.values, y_pred)
                    axes[0].set_title(u'Y与{}'.format(features))
                    idx = 1
                    # calculate with less features
                        for xab in combinations(features, len(features) - 1):
                            ...
                            # draw the ols-scatter
                            # axes[idx].plot(yab_test, yab_pred)
                            axes[idx].scatter(yab_test, yab_pred)
                            axes[idx].set_title(u'Y与{}'.format(xab))
                            idx += 1
                6) 应用模型: .predict(...[[x_n+1]]

            方法：
                1) 直接利用公式编写函数
                    解集 ws = (xMat.T * xMat).I * (xMat.T * yMat)
                    如果行列式为零，不可求逆：
                     if np.linalg.det(xMat.T * xMat) == 0.0: return None

                2) 利用现有的库
                    from sklearn.linear_model import LinearRegression
                    from sklearn.model_selection import train_test_split
                    from sklearn import metrics
                    import seaborn as sns


    MODIFIED  (MM/DD/YY)
        Na  12/25/2018

"""
__VERSION__ = "1.0.0.12252018"


# imports
import numpy as np
import pandas as pd
import os

# configuration
np.set_printoptions(precision=4, threshold=500)
np.random.seed(12345)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week10_data')
# CURRENT_PATH, DATA PATH here is for running code snippet in Python Console by Shift+Alt+E
# CURRENT_PATH = os.getcwd()
# DATA_PATH = os.path.join(CURRENT_PATH, r'pda\data\week10_data')

# functions
def main():
    # get data
    url1 = r'http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv'
    data = pd.read_csv(url1, index_col=0)
    columns = ['TV', 'Radio', 'Newspaper', 'Sales']
    data.columns = columns
    print('Sample data from {} head():\n{}'.format(url1, data.head()))
    print('\n\ttail():\n{}\n'.format(data.tail()))

    # draw scatter
    import seaborn as sns

    sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars=['Sales'], size=7, aspect=0.8)
    sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.8, kind='reg')

    # calculate corrcoef
    print('data corr_coef:\n{}\n'.format(data.corr()))

    # create X, Y datasets
    X = data[['TV', 'Radio', 'Newspaper']]
    print('X dataset head():\n{}'.format(X.head()))

    y = data['Sales']
    print('y dataset head():\n{}'.format(y.head()))

    # do regresssion analysis
    ## 直接根据系数矩阵公式计算
    def standRegres(xArr, yArr):
        xMat, yMat = np.mat(xArr), np.mat(yArr).T
        xTx = xMat.T * xMat
        # check if singular matrix
        ws = None
        if np.linalg.det(xTx) == 0.0:
            print('Matrix xArr is singular, cannot do inverse.\n\t{}\n\t...'.format(xMat.head(1)))
        else:
            ws = xTx.I * (xMat.T * yMat)
        return ws

    # add X_0 as intercept
    X2 = X
    X2['intercept'] = [1] * 200
    # 求解回归方程系数
    fitres = standRegres(X2, y)
    print('regression fit after manual calculation:\n\t{}'.format(fitres))

    ## 利用现有库求解
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()

    linreg.fit(X, y)
    print('regression fit result from sklearn.linear_model.LinearRegression():\n')
    print('\tintercept_:\t{}\n\tcoef:\t\t{}'.format(linreg.intercept_, linreg.coef_))
    print('Advertising channel -> coef:\n\t{}'.format(
        '\n\t'.join(['{} -> {:,.4f}'.format(k, v) for k, v in zip(columns[:3], linreg.coef_)])))

    # create train_set, test_set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print('After train_test_split:\n\tlen(X_train):\t{}\n\tlen(X_test):\t{}'.format(
        len(X_train), len(X_test)))

    linreg.fit(X_train, y_train)
    # result
    print('After train_test_split:\n\tintercept_:\t{}\n\tcoef_:\t\t{}'.format(
        linreg.intercept_, linreg.coef_))
    print('Advertising channel -> coef:\n\t{}\n'.format(
        '\n\t'.join(['{} -> {:,.4f}'.format(k, v) for k, v in zip(columns[:3], linreg.coef_)])))

    # predict
    y_pred = linreg.predict(X_test)
    N = 5
    print('predicted:\nX_test.head():\n{}\ny_pred[:{}]:\n{}'.format(
        X_test.head(), N, y_pred[:N]))

    # error evaluation
    from sklearn import metrics

    print('MAE: {}'.format(metrics.mean_absolute_error(y_test, y_pred)))
    print('MSE: {}'.format(metrics.mean_squared_error(y_test, y_pred)))
    print('NMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))

    # model evaluation
    feature_cols = columns[:2]
    X, y = data[feature_cols], data.Sales
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    print('After less features:\nMAE: {}'.format(metrics.mean_absolute_error(y_test, y_pred)))
    print('MSE: {}'.format(metrics.mean_squared_error(y_test, y_pred)))
    print('NMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))


# classes

# main entry
if __name__ == "__main__":
    main()
    