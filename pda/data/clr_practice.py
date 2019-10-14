#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        clr_practice.py

    DESCRIPTION
        分类：
        1) 定义：
            分类模型：输入样本的属性，输出对应的类别；将每个样本映射到预先定义好的类别。
        2）常用分类算法：
            KNN算法：  主要思想：K个，距离，投票决定多数类。
                        选取k个和待分类点距离最近的样本点；
                        看K个样本点的分类情况，投票决定待分类点所属的类。
                        <=> 类的区别比较大的时候，K可以取得小一点，否则大一点；
                            距离经常用欧式距离Euclidian distance，也可以用马氏距离Mahalanobis distance。

            决策树：    输入学习集，输出分类规则（决策树）。
                        主要思想：   空间划分


            逻辑回归

            贝叶斯分类器： 基于条件概率
                            p(ci|x1,x2,...,xn) = (p(x1|ci)p(x2|ci)...p(xn|ci)p(ci))/(p(x1)p(x2)...p(xn))

                            贝叶斯各个特征之间独立的强条件，限制了它的使用。
                            所以有了贝叶斯网络BaysianBeliefNetwork (BBN):
                                一个有向无环图，节点用点表示，依赖用边表示；
                                每个节点有一个条件概率表CPT，表明该节点和父母的条件概率。


            支持向量机SVM：
                原创性（非组合）的具有明显直观几何意义的分类算法，具有较高的准确率；
                思想直观、细节异常复杂，内容涉及凸分析算法、核函数、神经网络等领域。
                分两种情况：
                    简单情况：线性可分，把问题转化为一个凸优化问题，
                                可以用拉格朗日乘子法简化，然后用既有的算法解决。
                            如最优分割平面、最大边缘超平面（MMH）。

                    负责情况：线性不可分，用映射函数将样本投射到高维空间，使其变成线性可分的情形。
                                利用核函数来减少高纬度计算量。

            神经网络

        3） 其它知识点
        3.1） convert data to numbers
            Python分类中，准备数据的时候，可以用两种方法把字符型数据转换为数字：
            # Method1. manual mapping
            # None = 0, Basic = 1, Premium = 2
            data.loc[data.servicetype_bought == u'None', u'servicetype_bought'] = 0
            data.loc[data.servicetype_bought == u'Basic', u'servicetype_bought'] = 1
            data.loc[data.servicetype_bought == u'Premium', u'servicetype_bought'] = 2
            # or
            mapping = {u'source_website':  zip(('(direct)', 'digg', 'google', 'kiwitobes', 'slashdot'), range(5)),
                       u'region': zip(('UK', 'USA', 'France', 'New Zealand'), range(4)),
                       u'if_read_faq': zip(('yes', 'no'), range(2)),
                       u'servicetype_bought': zip(('None', 'Basic', 'Premium'), range(3))}
            for k, v in mapping.iteritems():
                data[k] = data[k].map({kk: vv for kk, vv in mapping[k]})

            # Method2. use LabelEncoder()
            from sklearn.preprocessing import LabelEncoder as LE
            le1 = LE()
            le.fit(['(direct)', 'digg', 'google', 'kiwitobes', 'slashdot'])
            data.source_website = le.fit_transform(data.source_website)

        3.2) difference between multiclass and multilabel in preparing data for classifier:
            Python分类中，对于 multiclass、multilabel 的场景，可以用不同的方法准备数据：
            #   multiclass 可用 LabelEncoder(), 或者 manual mapping
            #   multilabel 可用 LabelBinarizer()      —— 会把字符串当成一个 class,
            #                   MultiLabelBinarizer() —— 会把字符串中每一个字母当成一个 class

            from sklearn.preprocessing import MultiLabelBinarizer as MLB
            from sklearn.preprocessing import LabelBinarizer as LB

            lb = LB()
            lb.fit_transform(data.source_website)
            lb.classes_
            # <= array(['(direct)', 'digg', 'google', 'kiwitobes', 'slashdot'], dtype='|S9')

            mlb = MLB() # or mlb = MLB(classes=data.source_website.unique())
            mlb.fit_transform(data.source_website)
            mlb.classes_
            # <= array(['(', ')', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'o',
            #        'r', 's', 't', 'w'], dtype=object)


            get_dummies 可用于扩维，could be used in multilabel
                dummy_x = pd.get_dummies(x)

            # prepare x, y
            x = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            # y = data.iloc[:, 4].values.astype(np.int64)




    MODIFIED  (MM/DD/YY)
        Na  01/15/2019

"""
__VERSION__ = "1.0.0.01152019"


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split

# configuration
np.set_printoptions(precision=4, suppress=True, threshold=100)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format
plt.rc('figure', figsize=(10, 6))

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week13_data')
# CURRENT_PATH, DATA PATH here is for running code snippet in Python Console by ^+Alt+E
# CURRENT_PATH = os.getcwd()
# DATA_PATH = os.path.join(CURRENT_PATH, r'pda\data\week13_data')

# functions
def main():
    fsales = os.path.join(DATA_PATH, r'sales_data.xls')
    data = pd.read_excel(fsales, index_col=0)
    print("Original data example read from {}:\n{}".format(fsales, data.head()))

    # convert data from category to numbers
    # 数据是类别标签，要将它转换为数据
    # 用1来表示“好”、“是”、“高”这三个属性，用-1来表示“坏”、“否”、“低”
    data[data == u'好'] = 1
    data[data == u'是'] = 1
    data[data == u'高'] = 1
    data[data != 1] = -1
    # data.iloc[:, :3] is DataFrame,
    # data.iloc[:, :3].values.astype(int) is np.ndarray
    x = data.iloc[:, :3].values.astype(int)
    y = data.iloc[:, 3].values.astype(int)
    print("data got as x, y:\nx =\n{}\ny = \n{}".format(x, y))

    # split train_test_data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # KNN
    knnclf = KNC(algorithm='kd_tree')
    knnclf.fit(x_train, y_train)
    # use the model to predict
    knn_y_pred = knnclf.predict(x_test)
    print("\nx_test:\n{}\nknn_y_pred:\n{}\ny_test:\n{}".format(x_test, knn_y_pred, y_test))
    print("np.mean(knn_y_pred == y_test):\t{:,.4f}".format(np.mean(knn_y_pred == y_test)))
    print("knnclf.score(x_train, y_train):\t{:,.4f}".format(knnclf.score(x_train, y_train)))
    # validate: precision etc.
    precision, recall, thresholds = precision_recall_curve(y_train, knnclf.predict(x_train))
    print("KNN:\n\tprecision = {}\n\trecall = {}\n\tthresholds = {}".format(precision, recall, thresholds))
    print(u"clssification_report:\n{}".format(
        classification_report(y_test, knn_y_pred, target_names=[u'高', u'低'])))

    # Bayes
    bnbclf = BNB()
    bnbclf.fit(x_train, y_train)
    # use the model to predict
    bnb_y_pred = bnbclf.predict(x_test)
    print("\nx_test:\n{}\nbnb_y_pred:\n{}\ny_test:\n{}".format(x_test, bnb_y_pred, y_test))
    print("np.mean(bnb_y_pred == y_test):\t{:,.4f}".format(np.mean(bnb_y_pred == y_test)))
    print("bnbclf.score(x_train, y_train):\t{:,.4f}".format(bnbclf.score(x_train, y_train)))
    # validate: precision etc.
    precision, recall, thresholds = precision_recall_curve(y_train, bnbclf.predict(x_train))
    print("Bayes:\n\tprecision = {}\n\trecall = {}\n\tthresholds = {}".format(precision, recall, thresholds))
    print(u"clssification_report:\n{}".format(
        classification_report(y_test, bnb_y_pred, target_names=[u'高', u'低'])))

    # DTC
    dtcclf = DTC(criterion='entropy')
    dtcclf.fit(x_train, y_train)
    # save the DTC graphs
    # 导入相关函数，可视化决策树。
    # 导出的结果是一个dot文件，需要安装Graphviz才能将它转换为pdf或png等格式。
    # 然后我们可以使用的Graphviz的dot工具来创建一个PDF文件（或任何其他支持的文件类型）： dot -Tpdf iris.dot -o iris.pdf
        # 或者，如果我们安装了Python模块pydotplus，我们可以直接在Python中生成PDF文件（或任何其他支持的文件类型）：
        # >>> import pydotplus
        # >>> dot_data = tree.export_graphviz(clf, out_file=None)
        # >>> graph = pydotplus.graph_from_dot_data(dot_data)
        # >>> graph.write_pdf("iris.pdf")
    #   TODO
    # ref: http://cwiki.apachecn.org/pages/viewpage.action?pageId=10814387
    #
    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO
    ftree = os.path.join(DATA_PATH, r'tree.dot')
    with open(ftree, 'w') as f:
        f = export_graphviz(dtcclf, out_file=f)

    # use the model to predict
    dtc_y_pred = dtcclf.predict(x_test)
    print("\nx_test:\n{}\ndtc_y_pred:\n{}\ny_test:\n{}".format(x_test, dtc_y_pred, y_test))
    print("np.mean(dtc_y_pred == y_test):\t{:,.4f}".format(np.mean(dtc_y_pred == y_test)))
    print("dtcclf.score(x_train, y_train):\t{:,.4f}".format(dtcclf.score(x_train, y_train)))
    # validate: precision etc.
    precision, recall, thresholds = precision_recall_curve(y_train, dtcclf.predict(x_train))
    print("DecisionTreeClassifier:\n\tprecision = {}\n\trecall = {}\n\tthresholds = {}".format(precision, recall, thresholds))
    print(u"clssification_report:\n{}".format(
        classification_report(y_test, dtc_y_pred, target_names=[u'高', u'低'])))

    # SVM
    svcclf = SVC()
    svcclf.fit(x_train, y_train)
    # use the model to predict
    svc_y_pred = svcclf.predict(x_test)
    print("\nx_test:\n{}\nsvc_y_pred:\n{}\ny_test:\n{}".format(x_test, svc_y_pred, y_test))
    print("np.mean(svc_y_pred == y_test):\t{:,.4f}".format(np.mean(svc_y_pred == y_test)))
    print("svcclf.score(x_train, y_train):\t{:,.4f}".format(svcclf.score(x_train, y_train)))
    # validate: precision etc.
    precision, recall, thresholds = precision_recall_curve(y_train, svcclf.predict(x_train))
    print("SVM:\n\tprecision = {}\n\trecall = {}\n\tthresholds = {}".format(precision, recall, thresholds))
    print(u"clssification_report:\n{}".format(
        classification_report(y_test, svc_y_pred, target_names=[u'高', u'低'])))

# classes


# main entry
if __name__ == "__main__":
    main()
    