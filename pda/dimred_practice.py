#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        dimred_practice.py

    DESCRIPTION
        降维 Dimensionality Reduction
            1） 为什么需要降维：4点
                使得数据集容易使用；降低算法计算开销；去除噪声；使得结果易懂。

            2）降维方法
                主成分分析 PCA：坐标系变换。
                    Pearson于1901年提出，再由Hotelling（1933）加以发展的一种多变量统计方法；
                    通过析取主成分显出最大的个别差异，也用来削减回归分析和聚类分析中变量的数目；
                    可以使用样本协方差矩阵或相关系数矩阵作为出发点进行分析；
                    成分的保留：Kaiser主张（1960）将特征值小于1的成分放弃，只保留特征值大于1的成分；
                    如果能用不超过3-5个成分就能解释变异的80%，就算是成功。

                    基本思想：设法将原先众多具有一定相关性的指标，重新组合为一组新的互相独立的综合指标，并代替原先的指标。
                    通过对自变量进行线性变换，得到优化的指标；
                    把原先多个指标的计算降维为少量几个经过优化指标的计算（占去绝大部分份额）。

                    优点：降低数据的复杂度，识别最重要的多个特征；
                    缺点：不一定需要，且有可能损失有用信息。
                    适用数据类型：数值型数据。

                    PCA(n_components=None, copy=True, whiten=False,
                        svd_solver='auto', tol=0.0, iterated_power='auto',
                        random_state=None):
                        7 arguments.
                        属性：5个 components, explained_variance_ratio, mean_, n_components, noise_variance
                        方法：10个, fit(X[,y]), fit_transform(X[,y]), transform(X), inverse_transform(X);
                                get_covariance(), get_params([deep]), get_precision(), set_params(**params);
                                score(X[,y]), score_samples(X)


                因子分解：隐变量。
                    降维的一种方法，是主成分分析的推广和发展；
                    是用于分析隐藏在表面现象背后的因子作用的统计模型。
                    试图用最少个数的不可测的公共因子的线性函数与特殊因子之和来描述原来观测的每一分量。
                    例子：
                        1）各科学习成绩（数学能力，语言能力，运动能力等）
                        2）生活满意度（工作满意度、家庭满意度）
                    主要用途：3个
                        减少分析变量个数；使问题背后的业务因素的意义更加清晰呈现；
                        通过对变量间相关关系的探测，将原始变量分组，即将相关性高的变量分为一组，用公共因子来代替该变量。
                因子分析采用了更复杂的数学模型：
                    比主成分分析更加复杂的数学模型: X = u + AF + epsilon；
                        E(F)=0,Var(F)=Im,E(epsilon)=0,Var(epsilon)=D=diag(sigma1**2,...),Cov(F,epsilon)=0
                    求解模型的方法：主成分法、主因子法、极大似然法；
                    结果还可以通过因子旋转，使得业务意义更加明显。

                    其中，因子载荷矩阵是不唯一的，
                    因子载荷矩阵和特殊方差矩阵的估计：
                        主成分法： 通过样本估计期望和协方差矩阵；求协方差矩阵的特征值和特征向量；
                                    省去特征值较小的部分，求出A、D。

                        主因子法：首先对变量标准化；给出m和特殊方差的估计（初始）值；求出简约相关阵R*（p阶方阵）；
                                    计算R*的特征值和特征向量，取其前m个，略去其它部分；求出A*和D*，再迭代计算。

                        极大似然法：似然函数，极大似然函数。

                sklearn.decomposition.FactorAnalysis(n_components=None, tol=0.01, copy=True, max_iter=1000,
                    noise_variance_init=None, svd_method='randomized', iterated_power=3, random_state=0)：8arguments
                FactorAnalysis 属性：4个 components, loglike, noise_variance, n_iter;
                               方法：9个 fit, fit_transform, transform, get_covariance, get_params, get_precision,
                                         score, score_samples, set_params.


                主成分分析与因子分析的区别：
                    主成分分析侧重“变异量”，通过转换原始变量为新的组合变量使得数据的“变异量”最大，
                        从而能把样本个体之间的差异最大化，但得出来的主成分往往从业务场景的角度难以解释。
                    因子分析更重视相关变量的“共变异量”，组合的是相关性较强的原始变量，
                        目的是找到在背后起作用的少量关键因子，因子分析的结果往往更容易用业务知识去加以解释。

            3） 概念
                方差 ：      s_square = sum((xi - x_bar)**2) / (n-1)
                协方差：     cov(X, Y) = sum((xi - x_bar) (xi-y_bar) / (n-1)
                特征向量：   eigenvector 是一个矩阵，满足如下公式的非零向量：
                        A * v_bar = lambda * v_bar
                    其中v_bar是特征向量，A是方阵，lambda是特征值。

                SVD分解：  X = U * sum(V.T)

                两种方法：
                    方法1：对原数据去中心化，计算出新数据的协方差，或者计算原数据的相关系数（数据去不去中心化其相关系数不变），
                            再对协方差进行特征向量分解，取对应着特征值最大的特征向量。
                    方法2：直接使用原数据的SVD分解。




    MODIFIED  (MM/DD/YY)
        Na  01/29/2019

"""
__VERSION__ = "1.0.0.01292019"


# imports
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# configuration

# consts

# functions
def main():
    # load_iris
    data = load_iris()
    X, y = data.data, data.target
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X)

    # divide data based on reduced dimensionality
    red_x, red_y, blue_x, blue_y, green_x, green_y = [[] for _ in xrange(6)]
    for i in xrange(len(reduced_X)):
        if y[i] == 0:
            red_x.append(reduced_X[i][0])
            red_y.append(reduced_X[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_X[i][0])
            blue_y.append(reduced_X[i][1])
        else:
            green_x.append(reduced_X[i][0])
            green_y.append(reduced_X[i][1])

    # draw the scatter
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()

# classes

# main entry
if __name__ == "__main__":
    main()
    