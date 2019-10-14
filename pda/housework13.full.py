#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework_13.py

    DESCRIPTION
         my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]
        以上为某个网站的用户购买行为信息，第1列为来源网站，第2列为用户所在地区，
        第3列为是否阅读过FAQ，第4列为浏览网页数，第5列为购买的服务类型（目标变量）。
        通过构造合适的分类模型，预测用户最终的购买服务类型。

    MODIFIED  (MM/DD/YY)
        Na  01/15/2019

"""
__VERSION__ = "1.0.0.01152019"


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, collections
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report, precision_score, recall_score, \
    accuracy_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer as MLB
from sklearn.preprocessing import LabelBinarizer as LB
from sklearn.multiclass import OneVsRestClassifier as OvRC
from sklearn.multiclass import OneVsOneClassifier as OvOC
from sklearn.multiclass import OutputCodeClassifier as OCC
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB


# configuration
np.set_printoptions(precision=4, suppress=True, threshold=100)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format
plt.rc('figure', figsize=(10, 6))

# functions
def fit_a_model(estimator=None, x_train=None, y_train=None, x_test=None, y_test=None, target_names=None):
    if any(i is None for i in [estimator, x_train, y_train, x_test]):
        print('estimator, x_train, y_train, x_test, y_test are mandatory. Not all of them are given.\n'
              '=> Nothing is done. Exit...')
        return None
    else:
        pass
    # set up classifier
    clf = estimator.fit(x_train, y_train)
    # predict
    y_pred = clf.predict(x_test)
    # if y_test is given, calculate the score to evaluate the model
    clf_name = str(clf.__class__).split(r"'")[1]
    train_score, predict_score = clf.score(x_train, y_train), None
    print("{}:\n\ttrain score:  \t{:,.4f}".format(clf_name, train_score))
    # if y_test is not given, output the y_pred values without comparing it with y_test.
    if y_test is not None:
        predict_score = np.mean(y_pred == y_test)
        print("\tpredict score:\t{:,.4f}".format(predict_score))
        print("\tclassification_report:\n{}".format(classification_report(y_test, y_pred, target_names=target_names)))
    else:
        pass
    # output the predict result
    print("\n\ttrue value:\t\t{}\n\tpredict result:\t{}\n".format(
        y_test.values if y_test is not None else ['None']* len(x_test), y_pred))
    # return the model
    return clf, train_score, predict_score, y_pred

def main():
    # get data
    my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
               ['google', 'France', 'yes', 23, 'Premium'],
               ['digg', 'USA', 'yes', 24, 'Basic'],
               ['kiwitobes', 'France', 'yes', 23, 'Basic'],
               ['google', 'UK', 'no', 21, 'Premium'],
               ['(direct)', 'New Zealand', 'no', 12, 'None'],
               ['(direct)', 'UK', 'no', 21, 'Basic'],
               ['google', 'USA', 'no', 24, 'Premium'],
               ['slashdot', 'France', 'yes', 19, 'None'],
               ['digg', 'USA', 'no', 18, 'None'],
               ['google', 'UK', 'no', 18, 'None'],
               ['kiwitobes', 'UK', 'no', 19, 'None'],
               ['digg', 'New Zealand', 'yes', 12, 'Basic'],
               ['slashdot', 'UK', 'no', 21, 'None'],
               ['google', 'UK', 'yes', 18, 'Basic'],
               ['kiwitobes', 'France', 'yes', 19, 'Basic']]
    data = pd.DataFrame(data=my_data,
                        index=np.arange(len(my_data)),
                        columns=[u'source_website', u'region', u'if_read_faq', u'websites_read', u'servicetype_bought'])
    # convert data to numbers
    # Method1. manual mapping
    # # None = 0, Basic = 1, Premium = 2
    # data.loc[data.servicetype_bought == u'None', u'servicetype_bought'] = 0
    # data.loc[data.servicetype_bought == u'Basic', u'servicetype_bought'] = 1
    # data.loc[data.servicetype_bought == u'Premium', u'servicetype_bought'] = 2
    mapping = {u'source_website':  zip(('(direct)', 'digg', 'google', 'kiwitobes', 'slashdot'), range(5)),
               u'region': zip(('UK', 'USA', 'France', 'New Zealand'), range(4)),
               u'if_read_faq': zip(('yes', 'no'), range(2)),
               u'servicetype_bought': zip(('None', 'Basic', 'Premium'), range(3))}
    for k, v in mapping.iteritems():
        data[k] = data[k].map({kk: vv for kk, vv in mapping[k]})

    # # Method2. use LabelEncoder()
    # le1 = LE()
    # le.fit(['(direct)', 'digg', 'google', 'kiwitobes', 'slashdot'])
    # data.source_website = le.fit_transform(data.source_website)

    # # difference between multiclass and multilabel:
    # #   multiclass 可用 LabelEncoder(), 或者 manual mapping
    # #   multilabel 可用 LabelBinarizer()      —— 会把字符串当成一个 class,
    # #                   MultiLabelBinarizer() —— 会把字符串中每一个字母当成一个 class
    # lb = LB()
    # lb.fit_transform(data.source_website)
    # lb.classes_
    # # <= array(['(direct)', 'digg', 'google', 'kiwitobes', 'slashdot'], dtype='|S9')
    # mlb = MLB() # or mlb = MLB(classes=data.source_website.unique())
    # mlb.fit_transform(data.source_website)
    # mlb.classes_
    # # <= array(['(', ')', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'o',
    # #        'r', 's', 't', 'w'], dtype=object)


    #   get_dummies 可用于扩维，could be used in multilabel
    #               dummy_x = pd.get_dummies(x)

    # # # prepare x, y
    # x = data.iloc[:, :-1]
    # y = data.iloc[:, -1]
    # # y = data.iloc[:, 4].values.astype(np.int64)
    #
    # # # split train and test data
    # # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1, random_state=0)
    # # # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # # multiclass
    # # 50% etc. the numbers are for x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1, random_state=0)
    # # OneVsRestClassifier: linear=0, poly=50%, rbf=50%
    # ovrc_clf = fit_a_model(estimator=OvRC(SVC(random_state=0)),
    #                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                        target_names=[u'None', u'Basic', u'Premium'])
    #
    # # ovrc_clf_linear = fit_a_model(estimator=OvRC(SVC(kernel='rbf', probability=True, random_state=0)),
    # #                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    # #                        target_names=[u'None', u'Basic', u'Premium'])
    #
    # # ovrclf = OvRC(SVC(random_state=0)).fit(x_train, y_train)
    # # # ovrclf = OvRC(LinearSVC(random_state=0)).fit(x_train, y_train)
    # # ovrc_y_pred = ovrclf.predict(x_test)
    # # print("np.mean(knn_y_pred == y_test):\t{:,.4f}".format(np.mean(ovrc_y_pred == y_test)))
    # # print("ovrclf.score(x_train, y_train):\t{:,.4f}".format(ovrclf.score(x_train, y_train)))
    # # print("classification_report:\n{}".format(
    # #     classification_report(y_test, ovrc_y_pred, target_names=[u'None', u'Basic', u'Premium'])))
    # # print("OneVsRestClassifier: \n\ttest data:\t\t{}\n\tpredict result:\t{}".format(y_test.values, ovrc_y_pred))
    #
    # # OneVsOneClassifier : 50%
    # ovoc_clf = fit_a_model(estimator=OvOC(SVC(random_state=0)),
    #                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                        target_names=[u'None', u'Basic', u'Premium'])
    #
    # # ovoclf = OvOC(SVC(random_state=0)).fit(x_train, y_train)
    # # # ovoclf = OvOC(LinearSVC(random_state=0)).fit(x_train, y_train)
    # # ovoc_y_pred = ovoclf.predict(x_test)
    # # print("np.mean(knn_y_pred == y_test):\t{:,.4f}".format(np.mean(ovoc_y_pred == y_test)))
    # # print("ovrclf.score(x_train, y_train):\t{:,.4f}".format(ovoclf.score(x_train, y_train)))
    # # print("classification_report:\n{}".format(
    # #     classification_report(y_test, ovoc_y_pred, target_names=[u'None', u'Basic', u'Premium'])))
    # # print("OneVsRestClassifier: \n\ttest data:\t\t{}\n\tpredict result:\t{}".format(y_test.values, ovoc_y_pred))
    #
    # # [TO be removed] same as OneVsRestClassifier(SVC()) : 50%
    # ovoc_clf2 = fit_a_model(estimator=SVC(kernel='poly', probability=True, random_state=0),
    #                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                        target_names=[u'None', u'Basic', u'Premium'])
    #
    # # MultinomialNB : 25%
    # mnb_clf = fit_a_model(estimator=MNB(),
    #                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                        target_names=[u'None', u'Basic', u'Premium'])
    # # mnbclf = MNB().fit(x_train, y_train)
    # # mnb_y_pred = mnbclf.predict(x_test)
    # # print("np.mean(knn_y_pred == y_test):\t{:,.4f}".format(np.mean(mnb_y_pred == y_test)))
    # # print("ovrclf.score(x_train, y_train):\t{:,.4f}".format(mnbclf.score(x_train, y_train)))
    # # print("classification_report:\n{}".format(
    # #     classification_report(y_test, mnb_y_pred, target_names=[u'None', u'Basic', u'Premium'])))
    # # print("MultinomialNB: \n\ttest data:\t\t{}\n\tpredict result:\t{}".format(y_test.values, mnb_y_pred))
    #
    #
    #
    #
    # # KNN : 50%
    # knn_clf = fit_a_model(estimator=KNC(algorithm='kd_tree'),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    # # knnclf = KNC(algorithm='kd_tree', metric='minkowski')
    # # knnclf.fit(x_train, y_train)
    # # knn_y_pred = knnclf.predict(x_test)
    # # print("np.mean(knn_y_pred == y_test):\t{:,.4f}".format(np.mean(knn_y_pred == y_test)))
    # # print("knnclf.score(x_train, y_train):\t{:,.4f}".format(knnclf.score(x_train, y_train)))
    # # # validate: precision etc.
    # # #   precision_recall_curve works only for binary classifier, muliticlass not supported.
    # # #   multiclass could use precision_score(..., average='micro/macro'),
    # # #                        recall_score(..., average='micro/macro'),
    # # #   both could use classifcation_report, and it includes both micro and macro avg.
    # # #
    # # # for binary classifier
    # # # precision, recall, thresholds = precision_recall_curve(y_train, knnclf.predict(x_train))
    # # # for muliticlass
    # # # knn_y_train_pred = knnclf.predict(x_train)
    # # # precision = precision_score(y_train, knn_y_train_pred, average='micro')
    # # # recall = recall_score(y_train, knn_y_train_pred, average='micro')
    # # # accuracy = accuracy_score(y_train, knn_y_train_pred)
    # # # print("KNN:\n\tprecision = {:,.4f}\n\trecall = {:,.4f}\n\taccuracy = {:,.4f}".format(
    # # #     precision, recall, accuracy))
    # # print("classification_report:\n{}".format(
    # #     classification_report(y_test, knn_y_pred, target_names=[u'None', u'Basic', u'Premium'])))
    # # print("KNN: \n\ttest data:\t\t{}\n\tpredict result:\t{}".format(y_test.values, bnb_y_pred))
    # #
    # # # y_test_idx = 0
    # # # for idx in x_test.index:
    # # #     print("{:50}\t\t=> {}".format(my_data[idx], 'Basic' if knn_y_pred[y_test_idx] == 1 else 'Premium'))
    # # #     y_test_idx += 1
    #
    # # Bayes: 25%
    # bnb_clf = fit_a_model(estimator=BNB(),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    # # GNB: 25%
    # from sklearn.naive_bayes import GaussianNB
    # gnb_clf = fit_a_model(estimator=GaussianNB(),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    #
    #
    # # bnbclf = BNB()
    # # bnbclf.fit(x_train, y_train)
    # # # use the model to predict
    # # bnb_y_pred = bnbclf.predict(x_test)
    # # print("\nx_test:\n{}\nbnb_y_pred:\n{}\ny_test:\n{}".format(x_test, bnb_y_pred, y_test))
    # # print("np.mean(bnb_y_pred == y_test):\t{:,.4f}".format(np.mean(bnb_y_pred == y_test)))
    # # print("bnbclf.score(x_train, y_train):\t{:,.4f}".format(bnbclf.score(x_train, y_train)))
    # # # validate: clssification_report etc.
    # # print(u"clssification_report:\n{}".format(
    # #     classification_report(y_test, bnb_y_pred, target_names=[u'None', u'Basic', u'Premium'])))
    # # print("Bayes: \n\ttest data:\t\t{}\n\tpredict result:\t{}".format(y_test.values, knn_y_pred))
    # #
    # # # y_test_idx = 0
    # # # for idx in x_test.index:
    # # #     print("{:50}\t\t=> {}".format(my_data[idx], 'Basic' if bnb_y_pred[y_test_idx] == 1 else 'Premium'))
    # # #     y_test_idx += 1
    #
    #
    # # DTC: 50%
    # dtc_clf = fit_a_model(estimator=DTC(criterion='entropy'),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    # # ETC: 25%
    # from sklearn.tree import ExtraTreeClassifier
    # etc_clf = fit_a_model(estimator=ExtraTreeClassifier(),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    #
    # # dtcclf = DTC(criterion='entropy')
    # # dtcclf.fit(x_train, y_train)
    # # # use the model to predict
    # # dtc_y_pred = dtcclf.predict(x_test)
    # # print("dtcclf.predict_proba(x_test):\n{}\n".format(dtcclf.predict_proba(x_test)))
    # # print("\nx_test:\n{}\ndtc_y_pred:\n{}\ny_test:\n{}".format(x_test, dtc_y_pred, y_test))
    # # print("np.mean(dtc_y_pred == y_test):\t{:,.4f}".format(np.mean(dtc_y_pred == y_test)))
    # # print("dtcclf.score(x_train, y_train):\t{:,.4f}".format(dtcclf.score(x_train, y_train)))
    # # # validate: clssification_report etc.
    # # print(u"clssification_report:\n{}".format(
    # #     classification_report(y_test, dtc_y_pred, target_names=[u'None', u'Basic', u'Premium'])))
    # #
    # # # y_test_idx = 0
    # # # for idx in x_test.index:
    # # #     print("{:50}\t\t=> {}".format(my_data[idx], 'Basic' if dtc_y_pred[y_test_idx] == 1 else 'Premium'))
    # # #     y_test_idx += 1
    #
    #
    # # # SVM : multi_class='crammer_singer'=0,
    # # lsvc_clf = fit_a_model(estimator=LinearSVC(),
    # #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    # #                       target_names=[u'None', u'Basic', u'Premium'])
    #
    # # LR :  multi_class='multinomial', solver='saga' or 'sag' = 50%
    # #       multi_class='ovr' = 25%
    # from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    #
    # lr_clf = fit_a_model(estimator=LogisticRegression(multi_class='ovr'),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    #
    # # LRCV : multi_class='multinomial' = 50%, multi_class='ovr' = 50%
    # lrcv_clf = fit_a_model(estimator=LogisticRegressionCV(multi_class='ovr', random_state=0),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    #
    # # GPC : multi_class='one_vs_rest' = 25%, multi_class='one_vs_one' = 25%
    # from sklearn.gaussian_process import GaussianProcessClassifier
    # gpc_clf = fit_a_model(estimator=GaussianProcessClassifier(multi_class='one_vs_one'),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    #
    # # Perceptron : 50%
    # from sklearn.linear_model import Perceptron
    # gpc_clf = fit_a_model(estimator=Perceptron(),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    #
    # # SGD : 50%
    # from sklearn.linear_model import SGDClassifier
    # gpc_clf = fit_a_model(estimator=SGDClassifier(),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    #
    # # NuSVC : 50%
    # from sklearn.svm import NuSVC
    # gpc_clf = fit_a_model(estimator=NuSVC(),
    #                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    #                       target_names=[u'None', u'Basic', u'Premium'])
    #


    # 场景1：
    #   在用户购买信息的数据中，假定第5列为购买的服务类型（目标变量） == 'None' 为待购买服务的用户，用这些行作为测试集；
    #       其它行的数据为训练集。
    #   服务类型分为两种： 'Basic', 'Premium'，所以是二分类问题。
    #
    data_train, data_test = data[data.servicetype_bought != 0], data[data.servicetype_bought == 0]
    x_train, x_test = data_train.iloc[:, :-1], data_test.iloc[:, :-1]
    y_train, y_test = data_train.iloc[:, -1], data_test.iloc[:, -1]

    # binary classifier
    print(u"场景1：\n\t在用户购买信息的数据中，假定第5列为购买的服务类型（目标变量） == 'None' 为待购买服务的用户，"
          u"用这些行作为测试集；\n\t其它行的数据为训练集。\n\t"
          u"服务类型分为两种： 'Basic', 'Premium'，所以是二分类问题。\n")

    # KNN
    knn_clf = fit_a_model(estimator=KNC(algorithm='kd_tree'),
                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=None)
    # Bayes
    bnb_clf = fit_a_model(estimator=BNB(),
                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=None)
    # DTC
    dtc_clf = fit_a_model(estimator=DTC(criterion='entropy'),
                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=None)
    # LR
    lr_clf = fit_a_model(estimator=LogisticRegression(),
                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=None)
    # get the predict_result of the model which has the highest train_score,
    #   if more models take the first one
    model_data = pd.DataFrame(data=[knn_clf, bnb_clf, dtc_clf, lr_clf],
                              index = range(4),
                              columns=['model', 'train_score', 'predict_score', 'predict_result'])
    best_model = model_data.iloc[model_data.train_score.idxmax()]
    print(u"构造了{}种分类模型，最合适的模型为{}：\n\t训练得分：\t{:,.4f}\n\t预测结果：\t{}\n"
          u"=> 用户的最终购买服务类型预测为{}，即'{}'。\n".format(model_data.shape[0],
                                               str(best_model.model.__class__).split(r"'")[1].split(r'.')[-1],
                                               best_model.train_score,
                                               best_model.predict_result,
                                                 collections.Counter(best_model.predict_result).most_common(1)[0][0],
                                               'Basic' if best_model.predict_result.max() == 1 else 'Premium'))

    # best_model.predict_result.max(),
    # 场景2：
    #   用用户购买信息的数据中的70%用作训练集，30%作为测试集，预测用户的最终购买服务类型。
    #   服务类型分为三种： 'None', 'Basic', 'Premium'，多分类，所以用 multiclass 来构造分类模型。
    print(u"场景2：\n\t用用户购买信息的数据中的70%用作训练集，30%作为测试集，预测用户的最终购买服务类型。"
          u"服务类型分为三种： 'None', 'Basic', 'Premium'，多分类，所以用 multiclass 来构造分类模型。\n")

    # split train and test data
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # multiclass
    #  OneVsRestClassifier
    ovrc_clf = fit_a_model(estimator=OvRC(SVC(kernel='poly', random_state=0)),
                           x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                           target_names=[u'None', u'Basic', u'Premium'])

    # OneVsOneClassifier
    ovoc_clf = fit_a_model(estimator=OvOC(SVC(kernel='poly', random_state=0)),
                           x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                           target_names=[u'None', u'Basic', u'Premium'])
    # KNN
    knn_clf = fit_a_model(estimator=KNC(algorithm='kd_tree'),
                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                           target_names=[u'None', u'Basic', u'Premium'])

    # Bayes
    bnb_clf = fit_a_model(estimator=BNB(),
                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                           target_names=[u'None', u'Basic', u'Premium'])
    # DTC
    dtc_clf = fit_a_model(estimator=DTC(criterion='entropy'),
                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                           target_names=[u'None', u'Basic', u'Premium'])
    # LR
    lr_clf = fit_a_model(estimator=LogisticRegression(multi_class='ovr'),
                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                           target_names=[u'None', u'Basic', u'Premium'])
    # get the predict_result of the model which has the highest (predict_score, train_score),
    #   if more models take the first one
    model_data = pd.DataFrame(data=[ovrc_clf, ovoc_clf, knn_clf, bnb_clf, dtc_clf, lr_clf],
                              index=range(6),
                              columns=['model', 'train_score', 'predict_score', 'predict_result'])
    best_model = model_data.nlargest(1, ['predict_score', 'train_score'], keep='last').iloc[0]
    print(u"构造了{}种分类模型，最合适的模型为{}：\n\t预测得分：\t{}\n\t训练得分：\t{:,.4f}\n"
          u"=> 用户的最终购买服务类型预测为{}，即{}。\n".
          format(model_data.shape[0],
                 str(best_model.model.__class__).split(r"'")[1].split(r'.')[-1],
                 best_model.predict_score,
                 best_model.train_score,
                 best_model.predict_result,
                 ['None' if r==0 else('Basic' if r==1 else 'Premium') for r in best_model.predict_result]))


    # # 场景3：
    # #   用用户购买信息的数据除最后一行的数据用作训练集，用最后一行数据作为测试集，预测用户的最终购买服务类型。
    # #   服务类型分为三种： 'None', 'Basic', 'Premium'，多分类，所以用 multiclass 来构造分类模型。
    # # split train and test data
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1, random_state=0)
    # # multiclass
    # #  OneVsRestClassifier: linear=0, poly=50%, rbf=50%
    # ovrc_clf = fit_a_model(estimator=OvRC(SVC(random_state=0)),
    #                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    # # OneVsOneClassifier : 50%
    # ovoc_clf = fit_a_model(estimator=OvOC(SVC(random_state=0)),
    #                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    # # ......

    # TODO: which model is optimal


# main entry
if __name__ == "__main__":
    main()
    