#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        plot_practice.py

    DESCRIPTION
        plot_practice via matplotlib.pyplot, DataFrame.plot etc.
        1) matplotlib configuration:
           plt.rcParams: to check all current settings
           plt.rc(<group>, setting) to set, e.g. plt.rc('figure', figsize=(10, 6)), plt.rc('font', **font_options)
           # e.g.
            # font_options = {'family': 'monospace',
            #                 'weight': 'bold',
            #                 'size': 7}
            # plt.rc('font', **font_options)
            ## plt.rcdefaults()
            plt.rc('figure', figsize=(10, 6))
            # display in Chinese, ref: https://blog.csdn.net/tsinghuahui/article/details/73611128
            plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        2) show the image at non-interactive mode
                plt.show()
            # enable interactive mode
                plt.ion()
                ...
                plt.pause(5)
                plt.close('all')
            # ref: https://blog.csdn.net/xiaodongxiexie/article/details/78195860
            # ref: http://www.75271.com/14594.html
            # difference btn interactive mode and block mode https://blog.csdn.net/wonengguwozai/article/details/79686062

        3) Series.asof(where, subsets), DataFrame.asof(where, subsets):
            The last row without any NaN is taken
                (or the last row without NaN considering only the subset of columns in the case of a DataFrame)
            ref: https://blog.csdn.net/maymay_/article/details/80252587

        4) methods:
            fig = plt.figure(); ax = fig.add_subplot(111); ax = fig.add_subplot(211)
            fig, axes = plt.subplots(2，2)
            fig, (ax0, ax1) = plt.subplots(1, 2)
            fig.suptitle('2014 Sales Analysis', fontsize=14, fontweight='bold')

            plt.figure(1, figsize=(8, 8))
            # plt.axes: plt.axes((left, bottom, width, height), facecolor='w')

            plt.plot(np.random.randn(30).cumsum(), 'ko--', label=...), plt.bar(...), plt.barh(...), plt.scatter(...),
            n, nbins, patches = plt.hist(...), plt.boxplot(...)
            ax.annotate(label, xy=(date, spx.asof(date) + 50),
                    xytext=(date, spx.asof(date) + 200),
                    arrowprops=dict(facecolor='black'),
                    horizontalalignment='left', verticalalignment='top')
            ax.set_xlim(['1/1/2007', '1/1/2011']), ax.set_ylim([600, 1800])
            ax.set_xticks(...), ax.set_yticks(...); ax.settitle(...); ax.legend();
            ax.set_xticklabels(...), ax.set_yticklabels(...)
            ax.set_xlabel(...), ax.set_ylabel(...)
            ax.axvline(x=avg, color='b', label='Average', linestype='--', linewidth=1)
            ax.clear()

            plt.pie: Make a pie chart of array *x*.
                The fractional area of each wedge is given by ``x/sum(x)``.
                The wedges are plotted counterclockwise, by default starting from the x-axis.
                explode : specifies the fraction of the radius with which to offset each wedge.
                startangle : rotates the start of the pie chart by *angle* degrees counterclockwise from the x-axis.

            plt.pie(values, explode=explode, labels=labels,
                    autopct='%1.1f%%', startangle=97)
            plt.pie(values, explode=[0.2, 0.2, 0.2, 0.2], labels=labels,
                    autopct='%1.1f%%', startangle=67)
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=67)

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=None, hspace=None)
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            plt.plot(np.random.randn(30).cumsum(), 'ko--')
            plt.plot(np.random.randn(30).cumsum(), color='k', linestyle='dashed', marker='o')
            plt.plot(data, 'k-', drawstyle='steps-post', label='steps-post')
            plt.bar(x, y, yerr=xe, width=0.4, align='center',
                    ecolor='r', color='cyan', label='error bar experiment #1')
            # add numbers to bar via .text
                # 用 plt.text(...) 可以把数值写到每个柱上：
                #   其中，要显示的数值等于每个柱的高度height， x是要显示的数值的x坐标， y是是要显示的数值的y坐标。
                # ref: https://zhuanlan.zhihu.com/p/23636308
                # ref: https://blog.csdn.net/weixin_40198632/article/details/78858663
                fig = plt.figure(2)
                data = pd.Series(np.random.rand(10), index=list('abcdefghij'))
                bars = data.plot(kind='bar', title='data shown in bar_with_value', alpha=0.7)
                for p in bars.containers[0].patches:
                    height = p.get_height()
                    x, y = p.get_x() + p.get_width()/2 - 0.3, 1.02 * height
                    plt.text(x, y, '{:,.4f}'.format(height))

            rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
            circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
            pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                               color='g', alpha=0.5)

            for obj in (rect, circ, pgon):
                ax.add_patch(obj)

            plt.close('all')

            # fig.savefig
            ffig = os.path.join(DATA_PATH, r'figpath.svg')
            fig.savefig(ffig)
            fig.savefig(filename, dpi=400, bbox_inches='tight', ...)
            buffer=io.BytesIO();
            fig.savefig(buffer, format='png', ...), plt.savefig(...)
            plot_data = buffer.getvalue()

        5) write the image from saved-buffer to an embedded image of .html
            #   and check it over a browse via python -m SimpleHTTPServer 8000 &
            #   ref: https://stackoverflow.com/questions/14824522/dynamically-serving-a-matplotlib-image-to-the-web-using-python

        6) plotting in pd
            Series, DataFrame both has plot(kind=...) method,
                kind could be 10 types: line, bar, barh, box, scatter, hist, kde (=density), hexbin, area, pie
            df.plot() is line by default;

            pd.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)

        7) contour
            X, Y = np.meshgrid(x, y); Z = process_signals(X, Y)
            N = np.arange(-1, 1.5, 0.3)
            CS = plt.contour(Z, N, linewidths=2, cmap=mpl.cm.jet)
            plt.clabel(CS, inline=True, fmt='%1.1f', fontsize=10)
            plt.colorbar(CS)
            plt.title('My function: $z=(1-(x^2+y^2)) e^{-(y^3)/3}$')

        8) 3D bar, 3D hist
            from mpl_toolkits.mplot3d import Axes3D
            # mpl configuration
            mpl.rcParams['font.size'] = 10

            fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')

            # with 4 lines on z
            # for z in [2011, 2012, 2013, 2014]:
            #     xs = xrange(1, 13)
            #     ys = 1000 * np.random.rand(12)
            #     color = plt.cm.Set2(np.random.choice(xrange(plt.cm.Set2.N)))
            #     ax.bar(xs, ys, zs=z, zdir='ys', color=color, alpha=0.8)
            #
            # 为函数mpl_toolkits.mplot3d.Axes3D.plot指定xs、ys、zs和zdir参数。
            #   其他的参数则直接传给matplotlib.axes.Axes.plot。下面来解释一下这些特定的参数。
                # 1．xs和ys：x轴和y轴坐标。
                # 2．zs：这是z轴的坐标值，可以是所有点对应一个值，或者是每个点对应一个值。
                # 3．zdir：决定哪个坐标轴作为z轴的维度（通常是zs，但是也可以是xs或者ys）。
            # 作者：出版圈郭志敏
            # 链接：https://www.jianshu.com/p/bb8b25096df4
            # 來源：简书
            # 简书著作权归作者所有，简书任何形式的转载都请联系作者获得授权并注明出处。
            #
            # with 3 lines on zs: clearer
            # ref:  https://my.oschina.net/u/3942476/blog/2250732
            #
            # mpl.cm: 是matplotlib库中内置的色彩映射函数。
            # mpl.cm.[色彩]('[数据集]'): 即对[数据集]应用[色彩]
            # ref:  https://blog.csdn.net/baishuiniyaonulia/article/details/81416649
            #
            for z in [2015, 2016, 2017]:
                xs = xrange(1, 13)
                ys = 1000 * np.random.rand(12)
                color = plt.cm.Set2(np.random.choice(xrange(plt.cm.Set2.N)))
                ax.bar(xs, ys, zs=z, zdir='ys', color=color, alpha=0.8)

            ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xs))
            ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ys))
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            ax.set_zlabel('Sales Net [usd]')
            plt.show()

            # 3D hist
            fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
            samples = 25
            # data: x, y: normal distribution
            x, y = np.random.normal(5, 1, samples), np.random.normal(3, .5, samples)
            # xy平面上，按照10*10的网格划分，落在网格内个数hist，x划分边界、y划分边界
            hist, xedges, yedges = np.histogram2d(x, y, bins=10)
            elements = (len(xedges) - 1) * (len(yedges) - 1)
            # use xpos[0], ypos.T[0] could check the basic data used for broadcast
            xpos, ypos = np.meshgrid(xedges[:-1] + .25, yedges[:-1] + .25)
            xpos = xpos.flatten(); ypos = ypos.flatten()   # 多维数组变为一维数组
            zpos = np.zeros(elements)
            dx = .1 * np.ones_like(zpos)    # zpos一致的全1数组
            dy = dx.copy()
            dz = hist.flatten()

            # 每个立体以（xpos,ypos,zpos)为左下角，以（xpos+dx,ypos+dy,zpos+dz）为右上角
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', alpha=0.4)
            ax.set_xlabel('X Axis: Histogram2D Xedges')
            ax.set_ylabel('Y Axis: Histogram2D Yedges')
            ax.set_zlabel('Z Axis: Histogram2D hist\ni.e. numbers in grids')
            plt.show()
            plt.close('all')



    MODIFIED  (MM/DD/YY)
        Na  12/05/2018

"""
__VERSION__ = "1.0.0.12052018"


# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas import Series, DataFrame
import os, os.path
import pprint as pp

# configuration
# matplotlib configuration:
#   plt.rcParams: to check all current settings
#   plt.rc(<group>, setting) to set, e.g. plt.rc('figure', figsize=(10, 6)), plt.rc('font', **font_options)

# Setting font in matplotlib Method 1
# font_options = {'family': 'monospace',
#                 'weight': 'bold',
#                 'size': 7}
# plt.rc('font', **font_options)
#
# Setting font in matplotlib Method 2
# from matplotlib.font_manager import FontProperties
# fp = FontProperties(fname='name_of_ttc_file', size=10)
# plt.figure()
# plt.xlabel('xlabel1', fontproperties=fp)
# plt.title('title1', fontproperties=fp)
## ref: https://www.jianshu.com/p/738f6092ef5
#
## plt.rcdefaults()
plt.rc('figure', figsize=(10, 6))
# display in Chinese, ref: https://blog.csdn.net/tsinghuahui/article/details/73611128
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

np.set_printoptions(precision=4, threshold=500)
np.random.seed(12345)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format
# with pd.option_with('display.float_format', '{:,.4f}'.format');
# ref:  https://stackoverflow.com/questions/20937538/how-to-display-pandas-dataframe-of-floats-using-a-format-string-for-columns

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week7_data')
# CURRENT_PATH, DATA PATH here is for running code snippet in Python Console by ^+Alt+E
# CURRENT_PATH = os.getcwd()
# DATA_PATH = os.path.join(CURRENT_PATH, r'pda\data\week7_data')

# functions
def main():
    # plot via matplotlib
    # simple graphs
    plt.plot([1,2,3,2,3,2,2,1])
    # show the image at non-interactive mode
    plt.show()

    plt.plot([4,3,2,1],[1,2,3,4])
    plt.show()

    x = [1, 2, 3, 4]
    y = [5, 4, 3, 2]
    plt.figure()
    plt.subplot(231)
    plt.plot(x, y)

    plt.subplot(2, 3, 2)
    plt.bar(x, y)

    plt.subplot(233)
    plt.barh(x, y)

    plt.subplot(234)
    plt.bar(x, y)
    y1 = [7,8,5,3]
    plt.bar(x, y1, bottom=y, color='r')

    plt.subplot(235)
    plt.boxplot(x)

    plt.subplot(236)
    plt.scatter(x, y)

    plt.show()

    # figure and subplot
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    plt.show()

    plt.plot(np.random.randn(50).cumsum(), 'k--')
    fig.show()

    _ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
    ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
    plt.close('all')

    fig, axes = plt.subplots(2, 3)
    print('A series of axes generated via plt.subplots:\n{}'.format(axes))

    # plt.subplots_adjust(...)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=None)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for i in range(2):
        for j in range(2):
            _ = axes[i, j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)

    axes[0, 1].set_xticks(range(5))
    axes[1, 1].set_xticks(range(5, 10))
    plt.subplots_adjust(wspace=0, hspace=0)

    # matplotlib basic configuration for plot
    plt.figure()
    plt.plot(x, y, linestyle='--', color='g')
    plt.plot(np.random.randn(30).cumsum(), 'ko--')
    plt.plot(np.random.randn(30).cumsum(), color='k', linestyle='dashed', marker='o')
    plt.close('all')

    data = np.random.randn(30).cumsum()
    plt.plot(data, 'k--', label='Default')
    plt.plot(data, 'k-', drawstyle='steps-post', label='steps-post')
    plt.legend(loc='best')

    # set title, label, *ticks, *ticklabels
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.random.randn(1000).cumsum())

    ticks = ax.set_xticks([0, 250, 500, 750, 1000])
    labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                                rotation=30, fontsize='small')
    ax.set_title('My first matplotlib plot')
    ax.set_xlabel('Stages')
    plt.close('all')

    # add figures
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.plot(np.random.randn(1000).cumsum(), 'k', label='one')
    ax.plot(np.random.randn(1000).cumsum(), 'k--', label='two')
    ax.plot(np.random.randn(1000).cumsum(), 'k.', label='threee')
    ax.legend(loc='best')

    # annotate, add_patch
    #   Series.asof(where, subsets), DataFrame.asof(where, subsets):
    #       The last row without any NaN is taken
    #           (or the last row without NaN considering only the subset of columns in the case of a DataFrame)
    #       ref: https://blog.csdn.net/maymay_/article/details/80252587
    import datetime
    fig = plt.figure(); ax = fig.add_subplot(111)

    fcsv = os.path.join(DATA_PATH, r'spx.csv')
    data = pd.read_csv(fcsv, index_col=0, parse_dates=True)
    spx = data['SPX']
    spx.plot(ax=ax, style='k-')

    crisis_data = [
        (datetime.datetime(2007, 10, 11), 'Peak of bull market'),
        (datetime.datetime(2008, 3, 12), 'Bear Stearns Fails'),
        (datetime.datetime(2008, 9, 15), 'Lehman Bankruptcy')
    ]

    for date, label in crisis_data:
        ax.annotate(label, xy=(date, spx.asof(date) + 50),
                    xytext=(date, spx.asof(date) + 200),
                    arrowprops=dict(facecolor='black'),
                    horizontalalignment='left', verticalalignment='top')
    ax.set_xlim(['1/1/2007', '1/1/2011'])
    ax.set_ylim([600, 1800])
    ax.set_title('Important dates in 2008-2009 financial crisis')

    fig = plt.figure(); ax = fig.add_subplot(111)
    rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
    circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
    pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                       color='g', alpha=0.5)

    for obj in (rect, circ, pgon):
        ax.add_patch(obj)

    plt.close('all')

    # fig.savefig(filename, dpi=400, bbox_inches='tight', ...)
    ffig = os.path.join(DATA_PATH, r'figpath.svg')
    fig.savefig(ffig)
    ffig = os.path.join(DATA_PATH, r'figpath.png')
    fig.savefig(ffig, dpi=400, bbox_inches='tight')



    # fig.savefig(io.BytesIO(), format='png', ...), plt.savefig(...)
    from io import BytesIO
    buffer = BytesIO()
    # both work
    fig.savefig(buffer, format='png')
    # plt.savefig(buffer)
    print('{} Bytes data are saved to buffer: an io.BytesIO() object:\n{}'.format(
        buffer.__sizeof__(), buffer))

    plot_data = buffer.getvalue()
    print('{} Bytes data are read from saved buffer: an io.BytesIO() object:\n{}'.format(
        len(plot_data), plot_data))

    # write the image from saved-buffer to an embedded image of .html
    #   and check it over a browse via python -m SimpleHTTPServer 8000 &
    #   ref: https://stackoverflow.com/questions/14824522/dynamically-serving-a-matplotlib-image-to-the-web-using-python
    import base64
    img_str = "data:image/png;base64,"
    # base64.b64encode(buffer.getvalue()), buffer.getvalue().encode("base64").strip() are same.
    img_str = '{}{}'.format(img_str, base64.b64encode(buffer.getvalue()))

    fhtml = os.path.join(DATA_PATH, 'plot1.html')
    header = """<html><head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
        <title>1 png</title>
    </head><br>"""
    html_str = """<body>
    <h1>An example of .png generated via plt.plot and fig.savefig(BytesIO(), ...)</h1>
        <img src="%s"></img>
        <hr>
        <h3>to test if it works...<b>It works!</b>
            <br>ref:
            <a href="https://stackoverflow.com/questions/14824522/
            dynamically-serving-a-matplotlib-image-to-the-web-using-python">Dynamically serving a matplotlib 
            image to the web using python</a>
            <br><br>
        </h3>
    </body></html>
    """ % img_str
    with open(fhtml, 'w') as f:
        f.write('{}\n{}'.format(header, html_str))

    # run cmd in CLI: python -m SimpleHTTPServer 8000 &
    # in browser, open http://localhost:8000/pda/data/week7_data/plot1.html

    plt.close('all')

    # plotting in pd
    # line, df.plot() is line by default
    s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
    s.plot()

    df = DataFrame(np.random.randn(10, 4).cumsum(0),
                   columns=['A', 'B', 'C', 'D'],
                   index=np.arange(0, 100, 10))
    df.plot()
    df.plot(kind='line', subplots=True)

    # bar
    fig, axes = plt.subplots(2, 1)
    data = Series(np.random.rand(16), index=list('abcdefghijklmnop'))
    data.plot(kind='bar', ax=axes[0], color='r', alpha=0.7)
    data.plot(kind='barh', ax=axes[1], color='g', alpha=0.7)
    # add numbers to bar via .text
    # 用 plt.text(...) 可以把数值写到每个柱上：
    #   其中，要显示的数值等于每个柱的高度height， x是要显示的数值的x坐标， y是是要显示的数值的y坐标。
    # ref: https://zhuanlan.zhihu.com/p/23636308
    # ref: https://blog.csdn.net/weixin_40198632/article/details/78858663
    fig = plt.figure(2)
    data = pd.Series(np.random.rand(10), index=list('abcdefghij'))
    bars = data.plot(kind='bar', title='data shown in bar_with_value', alpha=0.7)
    for p in bars.containers[0].patches:
        height = p.get_height()
        x, y = p.get_x() + p.get_width()/2 - 0.3, 1.02 * height
        plt.text(x, y, '{:,.4f}'.format(height))

    df = DataFrame(np.random.rand(6, 4),
                   index=['one', 'two', 'three', 'four', 'five', 'six'],
                   columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
    df.plot(kind='bar')

    plt.figure()
    df.plot(kind='barh', stacked=True, alpha=0.5)
    plt.close('all')

    ftips = os.path.join(DATA_PATH, r'tips.csv')
    dtips = pd.read_csv(ftips)
    N = 10
    print('{} lines data read from {}. Top {}:\n{}'.format(
        dtips.size, ftips, N, dtips[:N]))
    party_counts = pd.crosstab(dtips.day, dtips['size'])
    print('pd.crosstab(...). data:\n{}'.format(party_counts))
    party_counts = party_counts.ix[:, 2:5]
    print('take part of the data:\n{}'.format(party_counts))

    party_pcts = party_counts.div(party_counts.sum(axis=1).astype(float), axis=0)
    print('convert to percentage:\n{}'.format(party_pcts))
    party_pcts.plot(kind='bar', stacked=True)

    # hist, kde
    print('tips data originally, top {}:\n{}'.format(N, dtips[:N]))
    dtips['tip_pct'] = dtips['tip'] / dtips['total_bill']
    print('tips data after added a new line tip_pct, top {}:\n{}'.format(N, dtips[:N]))

    plt.figure()
    dtips['tip_pct'].plot(kind='hist', bins=50)
    dtips['tip_pct'].plot(kind='kde')

    plt.figure()
    dtips['tip_pct'].hist(bins=50)

    comp1 = np.random.normal(0, 1, size=200)  # N(0, 1)
    comp2 = np.random.normal(10, 2, size=200)  # N(10, 4)
    values = Series(np.concatenate([comp1, comp2]))
    N = 10
    dfv = DataFrame(np.zeros((10, 2)))
    dfv[0], dfv[1] = values[:N].values, values[-N:].values
    dfv.columns = ['Top {}'.format(N), 'Bottom {}'.format(N)]
    print('Sample data from np.random.normal:\n{}'.format(dfv))

    plt.figure()
    values.hist(bins=100, alpha=0.3, color='k', normed=True)
    values.plot(kind='kde', style='k--')

    # scatter
    fmacro = os.path.join(DATA_PATH, r'macrodata.csv')
    dmacro = pd.read_csv(fmacro)
    data = dmacro[['cpi', 'm1', 'tbilrate', 'unemp']]
    print('Data read from {}. Top {}:\n{}'.format(fmacro, N, data[:N]))
    trans_data = np.log(data).diff().dropna()
    print('After np.log(data).diff().dropna(). Top {}:\n{}'.format(N, trans_data[:N]))

    plt.figure()
    plt.scatter(trans_data['m1'], trans_data['unemp'])
    plt.title('Changes in log {} vs {}'.format('m1', 'unemp'))

    pd.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)

    # matplotlib.plot
    # error bar
    x = np.arange(0, 10, 1); y = np.log(x)
    xe = 0.1 * np.abs(np.random.randn(len(y)))
    plt.figure()
    plt.bar(x, y, yerr=xe, width=0.4, align='center',
            ecolor='r', color='cyan', label='error bar experiment #1')
    plt.xlabel('# measurement'); plt.ylabel('Measured values')
    plt.title('Measurement'); plt.legend(loc='upper left')
    plt.show()
    plt.close('all')

    # pie
    #   plt.pie: Make a pie chart of array *x*.
    #           The fractional area of each wedge is given by ``x/sum(x)``.
    #           The wedges are plotted counterclockwise, by default starting from the x-axis.
    #           explode : specifies the fraction of the radius with which to offset each wedge.
    #           startangle : rotates the start of the pie chart by *angle* degrees counterclockwise from the x-axis.
    #
    #  plt.axes: plt.axes((left, bottom, width, height), facecolor='w')
    plt.figure(1, figsize=(8, 8))
    # ax=plt.axes([0.1, 0.1, 0.8, 0.8], facecolor='lightgreen')
    ax = plt.axes([0.1, 0.1, 0.8, 0.8])
    labels = ('Spring', 'Summer', 'Autumn', 'Winter')
    values = [15, 16, 16, 28]
    explode = [0.1, 0.1, 0.1, 0.1]

    # plt.pie(values, explode=explode, labels=labels,
    #         autopct='%1.1f%%', startangle=97)
    # plt.pie(values, explode=[0.2, 0.2, 0.2, 0.2], labels=labels,
    #         autopct='%1.1f%%', startangle=67)
    # plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=67)
    # ax.clear()
    plt.pie(values, explode=explode, labels=labels,
            autopct='%1.1f%%', startangle=67)
    plt.title('Rainy days by season')
    plt.show()
    plt.close('all')

    # contour
    import matplotlib as mpl

    def process_signals(x, y):
        return (1 - (x ** 2 + y ** 2)) * np.exp(-y ** 3 / 3)

    x = np.arange(-1.5, 1.5, 0.1)
    y = np.arange(-1.5, 1.5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = process_signals(X, Y)
    N = np.arange(-1, 1.5, 0.3)

    CS = plt.contour(Z, N, linewidths=2, cmap=mpl.cm.jet)
    plt.clabel(CS, inline=True, fmt='%1.1f', fontsize=10)
    plt.colorbar(CS)

    plt.title('My function: $z=(1-(x^2+y^2)) e^{-(y^3)/3}$')
    plt.show()

    # 3D bar
    from mpl_toolkits.mplot3d import Axes3D
    # mpl configuration
    mpl.rcParams['font.size'] = 10

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # with 4 lines on z
    # for z in [2011, 2012, 2013, 2014]:
    #     xs = xrange(1, 13)
    #     ys = 1000 * np.random.rand(12)
    #     color = plt.cm.Set2(np.random.choice(xrange(plt.cm.Set2.N)))
    #     ax.bar(xs, ys, zs=z, zdir='ys', color=color, alpha=0.8)
    #
    # 为函数mpl_toolkits.mplot3d.Axes3D.plot指定xs、ys、zs和zdir参数。
    #   其他的参数则直接传给matplotlib.axes.Axes.plot。下面来解释一下这些特定的参数。
        # 1．xs和ys：x轴和y轴坐标。
        # 2．zs：这是z轴的坐标值，可以是所有点对应一个值，或者是每个点对应一个值。
        # 3．zdir：决定哪个坐标轴作为z轴的维度（通常是zs，但是也可以是xs或者ys）。
    # 作者：出版圈郭志敏
    # 链接：https://www.jianshu.com/p/bb8b25096df4
    # 來源：简书
    # 简书著作权归作者所有，简书任何形式的转载都请联系作者获得授权并注明出处。
    #
    # with 3 lines on zs: clearer
    # ref:  https://my.oschina.net/u/3942476/blog/2250732
    #
    # mpl.cm: 是matplotlib库中内置的色彩映射函数。
    # mpl.cm.[色彩]('[数据集]'): 即对[数据集]应用[色彩]
    # ref:  https://blog.csdn.net/baishuiniyaonulia/article/details/81416649
    #
    for z in [2015, 2016, 2017]:
        xs = xrange(1, 13)
        ys = 1000 * np.random.rand(12)
        color = plt.cm.Set2(np.random.choice(xrange(plt.cm.Set2.N)))
        ax.bar(xs, ys, zs=z, zdir='ys', color=color, alpha=0.8)

    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xs))
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ys))
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_zlabel('Sales Net [usd]')
    plt.show()

    # 3D hist
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    samples = 25
    # data: x, y: normal distribution
    x, y = np.random.normal(5, 1, samples), np.random.normal(3, .5, samples)
    # xy平面上，按照10*10的网格划分，落在网格内个数hist，x划分边界、y划分边界
    hist, xedges, yedges = np.histogram2d(x, y, bins=10)
    elements = (len(xedges) - 1) * (len(yedges) - 1)
    # use xpos[0], ypos.T[0] could check the basic data used for broadcast
    xpos, ypos = np.meshgrid(xedges[:-1] + .25, yedges[:-1] + .25)

    xpos = xpos.flatten()   # 多维数组变为一维数组
    ypos = ypos.flatten()
    zpos = np.zeros(elements)

    dx = .1 * np.ones_like(zpos)    # zpos一致的全1数组
    dy = dx.copy()
    dz = hist.flatten()

    # 每个立体以（xpos,ypos,zpos)为左下角，以（xpos+dx,ypos+dy,zpos+dz）为右上角
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', alpha=0.4)
    ax.set_xlabel('X Axis: Histogram2D Xedges')
    ax.set_ylabel('Y Axis: Histogram2D Yedges')
    ax.set_zlabel('Z Axis: Histogram2D hist\ni.e. numbers in grids')

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.scatter(x, y)
    ax2.set_xlabel('X Axis: data in normal(5, 1)')
    ax2.set_ylabel('Y Axis: data in normal(3, .5)')
    plt.show()
    plt.close('all')



# classes

# main entry
if __name__ == "__main__":
    main()
    