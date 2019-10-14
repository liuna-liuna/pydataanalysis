#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        logisticreg_practice.py

    DESCRIPTION
        Logistic Regression 逻辑回归
        1. 特点
        逻辑回归是分类算法之一。
        广义线性回归的一种，把各独立自变量 Xn 经过线性回归模型映射到 z，再通过连续的 Sigmoid 函数映射到0~1；
            如果 P{Y=1} >=0.5，则分类为真，若 <0.5， 则为假。
            z：-∞ ~ +∞之间的连续变量；因变量 Y 是分类变量.
            z = θ₀ + θ₁ * X₁ + ... + θn * Xn = ln(P{Y=1} / P{Y=0}); 即： z = theta.T * X + b = w.T * X + b；
            y = P{Y=1} = P(y=1|X) = Sigma(z) = 1 / (1 + e-ᙆ)；
            p / (1-p) 称为事件的优势比 odds。
            对 odds 取自然对数即得 logistic 变换 logit(p) = y = ln(p/(1-p))。

            Sigmoid 函数输入： -∞ ~ +∞ , 输出： 0~1.

        逻辑回归可以直接计算：
                筛选数据；选定特征；
                初始化系数向量 w，如 w = 1；
                循环 maxloop ：计算梯度 = 对对数极大似然函数取w的偏导，
                              迭代解集 w：w = w + alpha * 梯度 = w + alpha *  x.T * error，
                              即用极大似然估计和梯度法计算w。
                由 z = ln(P{Y=1} / P{Y=0}) 计算 p，得出分类结果。
            也可以用 Python 模块计算：
                from sklearn.linear_model import LogisticRegression as LR
                from sklearn.linear_model import RandomizedLogisticRegression as RLR

        2. 回归的种类
        线性回归是拟合各点，逻辑回归是拟合两群或者N群点的分界面。
        线性回归的因变量 Y 是连续变量，自变量可以是连续变量、也可以是分类变量。
                 如果只有一个自变量且只有两类，该线性回归就等同于 t检验；
                 如果只有一个自变量且有三类或更多类，该线性回归就等同于方差分析；
                 如果有2个自变量，一个连续变量、一个分类变量，该线性回归就等同于协方差分析。
            其它条件，如独立性、线性、等方差性、正态性等。
        逻辑回归的因变量 Y 是分类变量，可以是二分类、也可以是多分类。

        Poission回归 用于服从 Poission分布的资料（Poission分布可以认为是计数资料）；
        Logistic回归 用于服从二项分布的资料 （二项分布可以认为是二分类数据）；
        负二项分布   用于负二项分布的资料（负二项分布也是个数，不过比 Poission分布更苛刻，个数+聚集性，可能就是负二项分布）。
            举例，如果调查流感的影响因素，结局当然是流感的例数，如果调查的人有的在同一个家庭里，由于流感具有传染性，
            那么同一个家里如果一个人得流感，那其他人可能也被传染，因此也得了流感，那这就是具有聚集性，
            这样的数据尽管结果是个数，但由于具有聚集性，因此用 poission 回归不一定合适，就可以考虑用负二项回归。

        作者：wyrover
        链接：https://www.jianshu.com/p/b17433c2ae00
        來源：简书
        简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。


        极大似然估计：就是让各个样本属于其真实标签的概率最大。是各个概率连乘，计算的时候经常取对数，称为对数极大似然函数。
        3. 逻辑回归为什么用极大似然估计：
                    逻辑回归输出的是概率，则考虑建立似然函数；
                    似然函数为连乘，则考虑取其对数似然；
                    目标是似然函数最大，则对似然函数取偏导，找到梯度方向；
                    用梯度方向和学习率，更新系数向量。

        4. 逻辑回归的步骤：4步：
            1） 设置并筛选特征；
            2） 列出回归方程，计算回归系数；
            3） 检验模型；
            4） 应用模型。



    MODIFIED  (MM/DD/YY)
        Na  01/02/2019

"""
__VERSION__ = "1.0.0.01022019"


# imports
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# configuration
plt.rc('figure', figsize=(10, 6))
np.random.seed(12345)
np.set_printoptions(precision=4, threshold=500)
pd.options.display.float_format = '{:,.4f}'.format
pd.options.display.max_rows = 100

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week11_data')
# CURRENT_PATH, DATA PATH here is for running code snippet in Python Console by ^+Alt+E
# CURRENT_PATH = os.getcwd()
# DATA_PATH = os.path.join(CURRENT_PATH, r'pda\data\week11_data')

# functions
def main():
    # Example 1 bankload.xls
    # read in data
    floan = os.path.join(DATA_PATH, r'bankloan.xls')
    data = pd.read_excel(floan)
    x = data.iloc[:, :8].as_matrix()
    y = data.iloc[:, 8].as_matrix()
    print("Sample data.head() from {}:\n{}".format(floan, data.head()))

    # imports
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.linear_model import RandomizedLogisticRegression as RLR

    # select features
    rlr = RLR()
    rlr.fit(x, y)
    fselected = rlr.get_support()
    print(u'Features selected via RandomizedLogisticRegression:\n\t{}'.format(
        ','.join(data.columns[:-1][fselected])))
    # list the lr equation, calculate the coef_
    x = data[data.columns[:-1][fselected]].as_matrix()
    lr = LR()
    lr.fit(x, y)
    # examine the model
    print('Score of the model via lr.score(x, y):\t{:,.4f}'.format(lr.score(x, y)))


    # Example 2 Sales x vs OpFee y
    # read in data
    x = pd.DataFrame([1.5, 2.8, 4.5, 7.5, 10.5, 13.5, 15.1, 16.5, 19.5, 22.5, 24.5, 26.5])
    y = pd.DataFrame([7.0, 5.5, 4.6, 3.6, 2.9, 2.7, 2.5, 2.4, 2.2, 2.1, 1.9, 1.8])

    # draw the scatter
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, y)
    fig.show()


    # linear regression
    from sklearn.linear_model import LinearRegression as LiR
    from sklearn.metrics import mean_squared_error

    linreg = LiR()
    linreg.fit(x, y)
    y_pred = linreg.predict(x)
    mse = mean_squared_error(y, y_pred)
    print("Linear model:\n\tcoef_:\t\t{}\n\tintercept:\t{}".format(linreg.coef_, linreg.intercept_))
    print("\tMSE:\t\t{:,.4f}".format(mse))
    # draw the fit curve
    ax.plot(x, y_pred, label='Linear model: mse={:,.4f}'.format(mse))
    ax.legend()


    # polynomial regression e.g. y = a + bx + cx^2
    px1, x2 = x.copy(), x**2
    px1['x2'] = x2
    plinreg = LiR()
    plinreg.fit(px1, y)
    py_pred = plinreg.predict(px1)
    pmse1 = mean_squared_error(y, py_pred)
    print("Polynomial model via X^2:\n\tcoef_:\t\t{}\n\tintercept:\t{}".format(plinreg.coef_, plinreg.intercept_))
    print("\tMSE:\t\t{:,.4f}".format(pmse1))
    # draw the fit curve
    ax.plot(x, py_pred, label='Polynomial model via $X^2$: mse={:,.4f}'.format(pmse1))
    ax.legend()


    # polynomial2
    from sklearn.preprocessing import PolynomialFeatures as PF
    pf = PF(degree=2)
    px2 = pf.fit_transform(x)
    plinreg2 = LiR()
    plinreg2.fit(px2, y)
    py_pred2 = plinreg2.predict(px2)
    pmse2 = mean_squared_error(y, py_pred2)
    print("Polynomial model via PolynomialFeatures:\n\tcoef_:\t\t{}\n\tintercept:\t{}".
          format(plinreg2.coef_, plinreg2.intercept_))
    print("\tMSE:\t\t{:,.4f}".format(pmse2))
    # draw the fit curve
    ax.plot(x, py_pred2, label='Polynomial model via PolynomialFeatures: mse={:,.4f}'.format(pmse2))
    ax.legend()

    # compare 2 polynomial methods
    #   np.allclose(a, b, rtol=1e-5, atol=1e-8, eqaul_nan)
    print("Two polynomial medels got same results? {}".format(np.allclose(py_pred, py_pred2)))

    # log, e.g. y = a + b logx
    logx = pd.DataFrame(np.log(x))
    loglinreg = LiR()
    loglinreg.fit(logx, y)
    logy_pred = loglinreg.predict(logx)
    logmse = mean_squared_error(y, logy_pred)
    print("Log model:\n\tcoef_:\t\t{}\n\tintercept:\t{}".format(loglinreg.coef_, loglinreg.intercept_))
    print("\tMSE:\t\t{:,.4f}".format(logmse))
    # draw the fit curve
    ax.plot(x, logy_pred, label='Log model: mse={:,.4f}'.format(logmse))
    ax.legend()


    # exponent, e.g. y = a e^(bx)
    expy = pd.DataFrame(np.log(y))
    explinreg = LiR()
    explinreg.fit(x, expy)
    expy_pred = explinreg.predict(x)
    expmse = mean_squared_error(expy, expy_pred)
    print("Exponent model:\n\tcoef_:\t\t{}\n\tintercept:\t{}".format(explinreg.coef_, explinreg.intercept_))
    print("\tMSE:\t\t{:,.4f}".format(expmse))
    # draw the fit curve
    ax.plot(x, np.exp(expy_pred), label='Exponent model: mse={:,.4f}'.format(expmse))
    ax.legend()


    # power, e.g. y = a x^b
    powlinreg = LiR()
    powlinreg.fit(logx, expy)
    powy_pred = powlinreg.predict(logx)
    powmse = mean_squared_error(expy, powy_pred)
    print("Power model:\n\tcoef_:\t\t{}\n\tintercept:\t{}".format(powlinreg.coef_, powlinreg.intercept_))
    print("\tMSE:\t\t{:,.4f}".format(powmse))
    # draw the fit curve
    ax.plot(x, np.exp(powy_pred), label='Power model: mse={:,.4f}'.format(powmse))
    ax.legend()

    # for non-interactive model
    plt.show()


    # Example 3. telco data
    fxls = os.path.join(DATA_PATH, r'telco.xls')
    data = pd.read_excel(fxls)
    print("Data in {}. data.describe():\n{}".format(fxls, data.describe()))
    x = data.iloc[:, :37].values
    y = data.iloc[:, 37].values
    # y = data.iloc[:, 37].as_matrix()  # string>:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.

    # LR model
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.metrics import mean_squared_error

    lr = LR()
    lr.fit(x, y)
    lry_pred = lr.predict(x)
    # score
    #  np.mean(y == y_pred) equals lr.score()
    lr_score = lr.score(x, y)
    lr_proba1 = lr.predict_proba(x)[:, 1]
    print("len(y_pred[y_pred == 1]) == len(lr_proba1[lr_proba1 >= 0.5])?\t{}".
          format(len(lry_pred[lry_pred == 1]) == len(lr_proba1[lr_proba1 >= 0.5])))
    print('LR model: score: {:,.4f}\tMSE: {:,.4f}'.format(lr_score, mean_squared_error(y, lry_pred)))
    # # plot, not necessary, better Confusion Matrix plot
    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(y, lry_pred, label='LR model scatter: score={:,.4f}'.format(lr_score))
    # ax.legend()
    # fig.show()

    # train_test_split
    from sklearn.model_selection import train_test_split
    p_testsize = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=p_testsize, random_state=1)
    # plot Confusion Matrix
    def cm_plot(y, yp):
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # draw the cm
        cm = confusion_matrix(y, yp)
        plt.matshow(cm, cmap=plt.cm.Greens)
        plt.colorbar()
        # annotate
        lencm = len(cm)
        for x in range(lencm):
            for y in range(lencm):
                plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', va='center')
        # labels
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        return plt

    # CART model
    # create model
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    # save the model
    treefile = os.path.join(DATA_PATH, r'tree.pkl')
    from sklearn.externals import joblib
    joblib.dump(tree, treefile)
    # cm_plot
    tt_y_pred = tree.predict(x_train)
    cm_plot(y_train, tt_y_pred).show()

    # plot ROC and AUC
    from sklearn.metrics import roc_curve, roc_auc_score
    tt_y_proba1 = tree.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, tt_y_proba1, pos_label=1)
    print('CART model:\n\tfpr={}\ttpr={}\tthresholds={}\n\troc_auc_score={:,.4f}'.
          format(fpr, tpr, thresholds, roc_auc_score(y_test, tt_y_proba1)))
    fig = plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label='ROC of CART', color='green')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.xlim(0, 1.05); plt.ylim(0, 1.05)
    plt.legend(loc=4)
    plt.show()

    # precision_recall_curve, classification_report
    from sklearn.metrics import precision_recall_curve, classification_report

    precision, recall, thresholds = precision_recall_curve(y_train, tt_y_pred)
    print('CART model:\n\tprecision={}\trecall={}\tthresholds={}\n'.format(precision, recall, thresholds))
    # precision, recall, f1-score, support
    print(classification_report(y_test, tree.predict(x_test), target_names=['High', 'Low']))

    # KNN model
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(algorithm='kd_tree')
    knn.fit(x_train, y_train)
    # results
    knny_pred = knn.predict(x_test)
    #  np.mean(y == y_pred) equals lr.score()
    print("np.mean(knny_pred == y_test):\t{}".format(np.mean(knny_pred == y_test)))
    precision, recall, thresholds = precision_recall_curve(y_train, knn.predict(x_train))
    print('KNN model:\n\tprecision={}\trecall={}\tthresholds={}\n'.format(precision, recall, thresholds))
    print(classification_report(y_test, knny_pred, target_names=['High', 'Low']))

    # Bayes NB model
    from sklearn.naive_bayes import BernoulliNB
    bnb = BernoulliNB()
    bnb.fit(x_train, y_train)
    # result
    bnby_pred = bnb.predict(x_test)
    print("np.mean(bnby_pred == y_test):\t{}".format(np.mean(bnby_pred == y_test)))
    precision, recall, thresholds = precision_recall_curve(y_train, bnb.predict(x_train))
    print('BernoulliNB model:\n\tprecision={}\trecall={}\tthresholds={}\n'.format(precision, recall, thresholds))
    print(classification_report(y_test, bnb.predict(x_test), target_names=['High', 'Low']))

    # SVM model
    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(x_train, y_train)
    # result
    svcy_pred = svc.predict(x_test)
    print("np.mean(svcy_pred == y_test):\t{}".format(np.mean(svcy_pred == y_test)))
    precision, recall, thresholds = precision_recall_curve(y_train, svc.predict(x_train))
    print('SVM model:\n\tprecision={}\trecall={}\tthresholds={}\n'.format(precision, recall, thresholds))
    print(classification_report(y_test, svc.predict(x_test), target_names=['High', 'Low']))






def self_sigmoid(x=None):
    import numpy as np
    # get input
    strx = raw_input(u'请输入一个整数或者小数，计算 Sigmoid 函数：')
    try:
        x = float(strx)
    except Exception as e:
        print(u'类型错误。请输入一个整数或者小数重试。')
        sys.exit(1)
    else:
        pass
    res = None
    # convert as sigmoid function: y = 1 / (1 + e^(-x))
    res = 1 / (1 + np.exp(-x))
    # if x == 0.0:
    #     res = 0.5
    # elif 0 < x <= sys.float_info.max:
    #     res = 1
    # else:
    #     res = 0
    # print output
    print(u'输入：\t\t\t{}\nSigmoid计算后：\t{}'.format(strx, res))
    return  res


# main entry
if __name__ == "__main__":
    main()
    