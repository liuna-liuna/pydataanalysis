#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        cluster_practice.py

    DESCRIPTION
        Clustering 聚类：
            非监督性学习，输入没有标签，即没有分类信息。
        3种方法：
            AgglomerativeClustering 凝聚聚类：8个参数
                AgglomerativeClustering(n_clusters=2, affinity='euclidean',
                                        memory=Memory(cachedir=None), connectivity=None, n_components=None,
                                        compute_full_tree='auto', linkage='ward', pooling_func=<function mean>)

                                        n_clusters, affinity, linkage 常用；
                                        connectivity：事先计算好的连接矩阵，即距离矩阵。
                                        n_components: 即将被舍去，算法自动计算。
                                        compute_full_tree: auto, True, False。
                                        pooling_func：计算类的平均值，很少用。

                属性：labels, n-leaves, n-components, children
                函数：fit, predict, get_params, set_params
                思想：4步
                    1）开始时，每个样本各自作为一类；
                    2）规定某种度量作为样本之间的距离及类与类之间的距离，并计算之；
                    3）将距离最短的两个类合并为一个新类；
                    4）重复2-3，即不断合并最近的两个类，每次减少一个类，直至所有样本被合并为一类。

            K-Means： 动态聚类, 10个参数
                KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)

                属性：cluster_centers, labels, inertia
                函数：fit, fit_predict, predict, get_params, set_params
                思想：4步
                    1）选择K个点作为初始质心；
                    2）将每个点指派到最近的质心，形成K个簇（聚类）；
                    3）重新计算每个簇的质心；
                    4）重复2-3直至质心不发生变化。
                肘部法则： TODO？
                K-means算法优缺点：
                    1）有效率，而且不容易受初始值选择的影响；
                    2）不能处理非球形的簇；
                    3）不能处理不同尺寸、不同密度的簇；
                    4）离群值可能有较大干扰（因此要先剔除）。

            Density-Based Spatial Clustering of Applications with Noise(DBScan): 7个参数
                本算法将具有 足够高密度 的区域划分为簇，并可以发现 任何形状 的聚类。
                概念：
                    r-邻域： 给定点半径r内的区域；
                    核心点： 如果一个点的r-邻域至少包含最少数目M个点，则称该点为核心点；
                    直接密度可达： 如果点p在核心点q的r-邻域内，则称p是从q出发可以直接密度可达；
                                    如果存在点链p1, p2, ..., pn, p1=q, pn=p, pi+1是从pi关于r和M直接密度可达，
                                        则称点p是从q关于r和M密度可达的；
                                    如果样本集D中存在点o， 使得点p、q是从o关于r和M密度可达的，
                                        那么点p、q是关于r和M密度相连的。
                DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto',
                        left_size=30, p=None, random_state=None)

                属性：core_sample_indices_, components_, labels_
                函数：fit, fit_predict, get_params, set_params
                思想：4步
                    1）指定合适的r和M；
                    2）计算所有的样本点，如果点p的r-邻域里有超过M个点，则创建一个以p为核心点的新簇；
                    3）反复寻找这些核心点直接密度可达（之后可能是密度可达）的点，将其加入到相应的簇，
                        对于核心点发生“密度相连”状况的簇，给予合并；
                    4）当没有新的点可以被添加到任何簇时，算法结束。
                算法描述：5步
                    输入：包含n个对象的数据库，半径e，最少数目MinPts；
                    输出：所有生成的簇，达到密度要求。
                    1）Repeat；
                    2）从数据库中抽出一个未处理的点；
                    3）IF 抽出的点是核心点 THEN 找出所有从该点密度可达的对象，形成一个簇；
                    4）ELSE 抽出的点是边缘点（非核心对象），跳出本次循环，寻找下一个点；
                    5）UNTIL 所有的点都被处理。
                DBSCAN对用户定义的参数很敏感，细微的不同都可能导致差别很大的结果；
                    而参数的选择无规律可循，只能靠经验确定。

        其中，3种样本点之间距离 affinity 的计算方法：
            欧式距离 euclidean： 通常意义下的距离。
            马氏距离 Mahalanobis：考虑到变量间的相关性，并且与变量的单位无关。
                                    主要是因为样本方差矩阵。
                                    如果是非圆形、非球形，用马氏距离更合适。

            余弦距离 cosine：     衡量变量的相似性。

        3种类与类之间距离 linkage 的计算方法：
            离差平方和 ward：
            类平均法 average：
            最大距离法 complete：

        # how to import train_test_split in sklearn with different version
        # ref: https://stackoverflow.com/questions/40704484/importerror-no-module-named-model-selection
        在 sklearn 0.18.0 之前，train_test_split 在模块 cross_validation 中，之后在 model_selection 中。

        from sklearn import __version__ as sklearn_version
        from distutils.version import LooseVersion
        if LooseVersion(sklearn_version) < '0.18.0':
            from sklearn.cross_validation import train_test_split
        else:
            from sklearn.model_selection import train_test_split



    MODIFIED  (MM/DD/YY)
        Na  01/23/2019

"""
__VERSION__ = "1.0.0.01232019"


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold, datasets
from scipy import ndimage

# configuration

# consts

# functions
def main():
    # data
    #  digits dataset:  Optical recognition of handwritten digits dataset
    # --------------------------------------------------
    # **Data Set Characteristics:**
    #     :Number of Instances: 5620
    #     :Number of Attributes: 64
    #     :Attribute Information: 8x8 image of integer pixels in the range 0..16.
    #     :Missing Attribute Values: None
    #     :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
    #     :Date: July; 1998
    #
    digits = datasets.load_digits(n_class=10)

    # visualize an image in digits
    plt.gray()
    plt.matshow(digits.images[0])
    plt.show()

    # get data
    K = 5
    X = digits.data
    y = digits.target
    n_samples, n_features = X.shape
    print("Sample data from datasets.load_digits:\n{}".format(X[:K, :]))
    print("n_samples = {}\tn_features = {}".format(n_samples, n_features))

    # Visualize the clustering
    def plot_clustering(X_red, X, labels, title=None, y=y, K=K):
        x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
        print("[In plot_clustering] before scaling:\n{}".format(X_red[:K]))
        X_red = (X_red - x_min) / (x_max - x_min)
        print("[In plot_clustering] after scaling:\n{}".format(X_red[:K]))

        plt.figure(figsize=(6, 4))
        for i in range(X_red.shape[0]):
            # plt.text(X_red[i, 0], X_red[i, 1], str(X[i]),
            #          color = plt.cm.spectral(labels[i] / 10.),
            #          fontdict = {'weight': 'bold', 'size': 9})

            plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                     color = plt.cm.nipy_spectral(labels[i] / 10.),
                     fontdict = {'weight': 'bold', 'size': 9})
            plt.xticks([])
            plt.yticks([])
            if title is not None:
                plt.title(title, size=17)
            else:
                pass
            plt.axis('off')
            plt.tight_layout()

    # 2D embedding of the digits dataset
    print('Computing embedding ...')
    print("[manifold.SpectralEmbedding] before scaling:\n{}".format(X[:K]))
    X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
    print("[manifold.SpectralEmbedding] after scaling:\n{}".format(X_red[:K]))
    print('Done.')

    # AgglomarativeClustering
    for linkage in ('ward', 'average', 'complete'):
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
        clustering.fit(X_red)
        plot_clustering(X_red, X, clustering.labels_, "{} linkage".format(linkage))

    plt.show()

    # draw data in plot
    def create_ax():
        fig, ax = plt.subplots(1, 1)
        ax.axis([-1, 9, -1, 9])
        ax.grid(True)
        return  ax

    X0 = np.array([7, 5, 7, 3, 4, 1, 0, 2, 8, 6, 5, 3])
    X1 = np.array([5, 7, 7, 3, 6, 4, 0, 2, 7, 8, 5, 7])
    ax = create_ax()
    ax.plot(X0, X1, 'k.')

    C1 = [1, 4, 5, 9, 11]
    C2 = list(set(range(12)) - set(C1))
    X0C1, X1C1 = X0[C1], X1[C1]
    X0C2, X1C2 = X0[C2], X1[C2]
    ax = create_ax()
    ax.plot(X0C1, X1C1, 'rx')
    ax.plot(X0C2, X1C2, 'g.')
    ax.plot(4, 6, 'rx', ms=12.0)
    ax.plot(5, 5, 'g.', ms=12.0)

    C1 = [1, 2, 4, 8, 9, 11]
    C2 = list(set(range(12)) - set(C1))
    X0C1, X1C1 = X0[C1], X1[C1]
    X0C2, X1C2 = X0[C2], X1[C2]
    ax = create_ax()
    ax.plot(X0C1, X1C1, 'rx')
    ax.plot(X0C2, X1C2, 'g.')
    ax.plot(3.8, 6.4, 'rx', ms=12.0)
    ax.plot(4.57, 4.14, 'g.', ms=12.0)

    C1 = [0, 1, 2, 4, 8, 9, 10, 11]
    C2 = list(set(range(12)) - set(C1))
    X0C1, X1C1 = X0[C1], X1[C1]
    X0C2, X1C2 = X0[C2], X1[C2]
    ax = create_ax()
    ax.plot(X0C1, X1C1, 'rx')
    ax.plot(X0C2, X1C2, 'g.')
    ax.plot(5.5, 7.0, 'rx', ms=12.0)
    ax.plot(2.2, 2.8, 'g.', ms=12.0)


    # TODO
    pass


# main entry
if __name__ == "__main__":
    main()
    