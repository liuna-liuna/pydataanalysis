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
import os, collections
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier as OvRC
from sklearn.multiclass import OneVsOneClassifier as OvOC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# configuration
np.set_printoptions(precision=4, suppress=True, threshold=100)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format

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
    mapping = {u'source_website':  zip(('(direct)', 'digg', 'google', 'kiwitobes', 'slashdot'), range(5)),
               u'region': zip(('UK', 'USA', 'France', 'New Zealand'), range(4)),
               u'if_read_faq': zip(('yes', 'no'), range(2)),
               u'servicetype_bought': zip(('None', 'Basic', 'Premium'), range(3))}
    for k, v in mapping.iteritems():
        data[k] = data[k].map({kk: vv for kk, vv in mapping[k]})

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
    lr_clf = fit_a_model(estimator=LogisticRegression(solver='lbfgs'),
                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=None)
    # get the predict_result of the model which has the highest train_score,
    #   if more models take the first one
    model_data = pd.DataFrame(data=[knn_clf, bnb_clf, dtc_clf, lr_clf],
                              index = range(4),
                              columns=['model', 'train_score', 'predict_score', 'predict_result'])
    best_model = model_data.iloc[model_data.train_score.idxmax()]
    print(u"构造了{}种分类模型，最合适的模型为{}：\n\t训练得分：\t{:,.4f}\n\t预测结果：\t{}\n"
          u"=> 用户的最终购买服务类型预测为{}，即'{}'。\n".
          format(model_data.shape[0],
                 str(best_model.model.__class__).split(r"'")[1].split(r'.')[-1],
                 best_model.train_score,
                 best_model.predict_result,
                 collections.Counter(best_model.predict_result).most_common(1)[0][0],
                 'Basic' if best_model.predict_result.max() == 1 else 'Premium'))


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
    ovrc_clf = fit_a_model(estimator=OvRC(SVC(kernel='poly', gamma='auto', random_state=0)),
                           x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                           target_names=[u'None', u'Basic', u'Premium'])

    # OneVsOneClassifier
    ovoc_clf = fit_a_model(estimator=OvOC(SVC(kernel='poly', gamma='auto', random_state=0)),
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
    lr_clf = fit_a_model(estimator=LogisticRegression(multi_class='ovr', solver='lbfgs'),
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


# main entry
if __name__ == "__main__":
    main()
    