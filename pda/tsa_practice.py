#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        tsa_practice.py

    DESCRIPTION
        Time Series Analysis时间序列分析：
        1）一种动态数据处理的统计方法，基于随机过程理论 和 数理统计学方法，
            研究随机数据序列所遵从的统计规律，以用于解决实际问题。
        2）时间序列算法：平滑法（用平移的方法消除短期的波动）、趋势拟合法、组合模型、
                        AR模型(AutoRegression，和回归方程式类似)、MA模型(MovingAverage，关于残差的方程式)、
                        ARMA模型、ARIMA模型（差分自动回归移动平均模型）、
                        ARCH模型（针对自相关很强的数据）、GARCH模型（General ARCH模型）及其衍生模型。
        3）Python中的时间：
            模块：datetime, time, calendar;
            类型：date（年月日）, time（时分秒毫秒）, datetime, timedelta（日秒毫秒）.
            字符串和 datetime 互换：格式码：%Y-%m-%d...
                                    dateutil.parser.parse('Jan 31, 1997 10:45 PM', dayfirst=False, ...)
                                    idx = pd.to_datetime(datestrs + [None]), pd.isnull(idx) ...

        4) Python中的时间序列类型：以时间戳为索引的Series
            索引： 格式可以是字符串、datetime对象、DatetimeIndex对象,
                    ts.index[1], ts['1/10/2011'], ts['20110110']

            选取：下标、切片

            子集构造：切片,
                longer_ts['2001'], long_ts['2001-05'], ts['datetime(2011, 1, 7):], ts[::2],
                ts.truncate(after='1/9/2011')

            生成日期范围:
                pd.date_range(start=..., end=..., periods=..., freq=...)

            频率与日期偏移量

            移动数据
        5） 时间序列绘图：
            移动窗口与指数加权函数:  14 s1/df1.rolling(data, expand_window, ...).mean() ... in pandas 0.23.4
                                            instead of rolling_mean
                                   5 s1/df1.ewm(data, expand_window, ...).mean() ...
                                            instead of ewma, ewmvar, ewmstd, ewmcorr, ewmcov.
        6）时间序列预处理
            平稳性检验：
                平稳性：假定时间序列每一个数值都是从一个概率分布中随机得到，如果满足3条件：
                        1）均值：与时间t无关的常数；
                        2）方差：与时间t无关的常数；
                        3）协方差：只与时间间隔k有关，与时间t无关的常数。
                        则称该随机事件序列是平稳的 stationary。

                平稳性检验：
					图检验：	时序图检验、自相关检验；操作简单、应用广泛但带有主观性。
					假设检验：	单位根检验；需要构造检验统计量。

            纯随机性检验：
                白噪声：	纯随机过程。
                白噪声检验：Q统计量、LB统计量。

		7） 平稳时间序列分析
			例如3个模型：
			
				自回归模型AR
				滑动平均模型MA
				自回归滑动平均模型ARMA
				
			流程：7 steps：
				平稳非白噪声序列 -> 计算ACF、PACF -> ARMA模型识别
				 -> 估计模型中未知参数 -> 模型检验 -> 模型优化
				 -> 预测未来的走势
			


    MODIFIED  (MM/DD/YY)
        Na  01/11/2019

"""
__VERSION__ = "1.0.0.01112019"


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

# configuration
np.set_printoptions(precision=4, threshold=500, suppress=True)
pd.options.display.max_rows = 500
pd.options.display.float_format = '{:,.4f}'.format
plt.rc('figure', figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# to suppress Pandas Future warning
#   ref: https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week12_data')
# CURRENT_PATH, DATA PATH here is for running code snippet in Python Console by ^+Alt+E
# CURRENT_PATH = os.getcwd()
# DATA_PATH = os.path.join(CURRENT_PATH, r'pda\data\week12_data')

# functions
def main():
    import datetime as dt
    ndt = dt.datetime.now()
    print(u'Current timestamp via dt.datetime.now: {}年{}月{}日{}时{}分{}秒'.
          format(ndt.year, ndt.month, ndt.day, ndt.hour, ndt.minute, ndt.second))
    delta = dt.datetime(2011, 1, 7) - dt.datetime(2008, 6, 24, 8, 15)
    print('delta: {} days {} seconds'.format(delta.days, delta.seconds))

    start = dt.datetime(2011, 1, 7)
    print('start + dt.timedelta(12):\t{}'.format(start + dt.timedelta(12)))
    print('start - dt.timedelta(12):\t{}'.format(start - dt.timedelta(12)))

    # string -> datetime
    stamp = dt.datetime(2011, 1, 3)
    print("datetime -> string:\nstr(stamp): {}\tstamp.strftime('%Y-%m-%d'): {}".format(
        str(stamp), stamp.strftime('%Y-%m-%d')))

    value = '2011-01-03'
    print("string -> datetime:\n\tdt.datetime.strptime(value, '%Y-%m-%d'): {}".format(
        dt.datetime.strptime(value, '%Y-%m-%d')))
    datestrs = ['7/6/2011', '8/6/2011']
    print([dt.datetime.strptime(x, '%m/%d/%Y') for x in datestrs])

    import dateutil.parser as parser
    print("dateutil.parser:\n\tparse('2011-01-03'): {}".format(parser.parse('2011-01-03')))
    print("dateutil.parser:\n\tparse('Jan 31, 1997 10:45 PM'): {}".format(parser.parse('Jan 31, 1997 10:45 PM')))
    print("dateutil.parser:\n\tparse('6/12/2011', dayfirst=True): {}".format(parser.parse('6/12/2011', dayfirst=True)))

    print("\npd.to_datetime(datestrs):\n{}".format(pd.to_datetime(datestrs)))
    idx = pd.to_datetime(datestrs + [None])
    print("pd.to_datetime(datestrs + [None]):\n{}".format(idx))
    print("pd.isnull(idx):\t{}".format(pd.isnull(idx)))

    # Time Series in pandas
    from datetime import datetime
    from pandas import Series, DataFrame
    dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
             datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
    ts = Series(np.random.randn(6), index=dates)
    print("\ntype(ts): {}\nts.index: {}\nts.index.dtype: {}".format(
        type(ts), ts.index, ts.index.dtype))
    stamp = ts.index[0]
    print("stamp = ts.index[0] = {}".format(stamp))

    # index, selection and subsets construct
    stamp = ts.index[2]
    print("stamp = {}\t\t\tts[stamp] = {:,.4f}".format(stamp, ts[stamp]))
    print("Time Series data:\n{}".format(ts))
    print("\tts['1/10/2011']: {:,.4f}".format(ts['1/10/2011']))
    print("\tts['20110110']: {:,.4f}".format(ts['20110110']))

    K = 1000
    longer_ts = Series(np.random.randn(K),
                       index=pd.date_range('1/1/2000', periods=K))
    print("longer_ts:\n{}".format(longer_ts))
    print("longer_ts['2001']:\n{}".format(longer_ts['2001']))
    print("longer_ts['2001-05']:\n{}".format(longer_ts['2001-05']))
    print("ts[datetime(2011, 1, 7):]:\n{}".format(ts[datetime(2011, 1, 7):]))
    print("ts['1/6/2011': '1/11/2011']:\n{}".format(ts['1/6/2011': '1/11/2011']))
    print("ts.truncate(after='1/9/2011'):\n{}".format(ts.truncate(after='1/9/2011')))

    nrows, ncols = 100, 4
    dates = pd.date_range('1/1/2000', periods=nrows, freq='W-WED')
    long_df = DataFrame(np.random.randn(nrows, ncols),
                        index=dates,
                        columns=['Colorado', 'Texus', 'New York', 'Ohio'])
    # long_df.ix['2001-5']m long_df.ix['05/2001'], long_df.ix['2001/5'] work also.
    print("long_df.ix['5-2001']:\n{}".format(long_df.ix['5-2001']))

    # when index is duplicated DatetimeIndex, use .groupby(level=0) to get grouped value by dates:
    dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000', '1/3/2000'])
    dup_ts = Series(np.arange(5), index=dates)
    print("dup_ts:\n{}".format(dup_ts))
    print("dup_ts.index.is_unique:\t{}".format(dup_ts.index.is_unique))
    print("dup_ts['1/3/2000']:\n\t{}".format(dup_ts['1/3/2000']))
    print("dup_ts['1/2/2000']:\n{}".format(dup_ts['1/2/2000']))

    grouped = dup_ts.groupby(level=0)
    print("dup_ts.index:\n{}".format(dup_ts.index))
    print("dup_ts.groupby(level=0).mean():\n{}".format(grouped.mean()))
    print("dup_ts.groupby(level=0).count():\n{}".format(grouped.count()))

    # date_range, frequency, shift
    #   from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd
    #   pd.date_range(start=..., end=...., freq=..., periods=...)
    #       freq中偏移量的别名：
    #           D: Day, B: BusinessDay, H: Hour, T/min: Minutes, S: Seocnds,
    #           L: Milli毫秒, U: Micro微秒,
    #           M: MonthEnd, BM: Business MonthEnd, MS: MonthStart, BMS: BusinessMonthStart,
    #           W-MON, W-TUE: Weekly, WOM-1MON, WOM-2MON: the 1st MON in WeekOfMonth,
    #           Q-JAN, Q-FEB: The last CalenderDay of Quarter which ends with JAN, FEB,
    #           BQ-JAN, BQ-FEB: Business Q-JAN, Business Q-FEB,
    #           QS-JAN, QS-FEB: The first CalenderDay of Quarter which ends with JAN, FEB,
    #           BQS-JAN, BQS-FEB: BusinessQuarterBegin
    #           A-JAN, A-FEB: YearEnd, the last CalenderDay in JAN, FEB in each year
    #           BA-JAN, BA-FEB: BusinessYearEnd,
    #           AS-JAN, AS-FEB: YearBegin, BAS-JAN, BAS-FEB: BusinessYearBegin.
    #
    print("ts.resample('D').mean:\n{}".format(ts.resample('D').mean()))
    index = pd.date_range('4/1/2012', '6/1/2012')
    print("pd.date_range examples:")
    print("pd.date_range('4/1/2012', '6/1/2012'):\n{}".format(index))
    pd.date_range(start='4/1/2012', periods=20)
    pd.date_range(end='6/1/2012', periods=20)
    pd.date_range('4/1/2012', '6/1/2012', freq='BM')
    pd.date_range('5/2/2012 12:56:31', periods=5)
    pd.date_range('5/2/2012 12:56:31', periods=5, normalize=True)

    from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd
    hour = Hour()
    print("in pandas.tseries.offsets: {}".format(hour))
    four_hour = Hour(4)
    print("4hours: {}".format(four_hour))

    print("freq='4h':\n{}".format(pd.date_range('1/1/2000', '1/3/2000 23:59', freq='4h')))
    print("2.5hours: {}".format(Hour(2) + Minute(30)))
    print("freq=1h30min:\n{}".format(pd.date_range('1/1/2000', periods=10, freq='1h30min')))
    idx = pd.date_range('1/1/2012', '9/1/2012', freq='WOM-3FRI')
    print("WOM-3FRI:\n{}".format(idx))
    print("DatetimeIndex -> list:\n{}".format(list(idx)))

    K = 4
    ts = Series(np.random.randn(K),
                index=pd.date_range('1/1/2000', periods=4, freq='M'))
    print("ts with freq='M':\n{}".format(ts))

    print("ts before shift:\n{}".format(ts))
    print("ts after shift(2):\n{}".format(ts.shift(2)))
    print("ts after shift(-2):\n{}".format(ts.shift(-2)))
    print("returns calculated manually:\n{}".format(ts / ts.shift(1) - 1))
    print("ts.shift(2, freq='M'):\n{}".format(ts.shift(2, freq='M')))
    print("ts.shift(3, freq='D'):\n{}".format(ts.shift(3, freq='D')))
    print("ts.shift(1, freq='3D'):\n{}".format(ts.shift(1, freq='3D')))
    print("ts.shift(1, freq='90T'):\n{}".format(ts.shift(1, freq='90T')))

    #  3 Methods to get the MonthEnd date
    #   cdate + MonthEnd(K); MonthEnd().rollforward(cdate) / .rollback(cdate);
    #   pd.date_range(..., freq='M');
    #
    cdate = datetime(2011, 11, 17)
    print("current date:\t\t{}".format(cdate))
    print("current date + 3 * Day():\t\t{}".format(cdate + 3 * Day()))
    print("current date + MonthEnd():\t\t{}".format(cdate + MonthEnd()))
    print("current date + MonthEnd(2):\t\t{}".format(cdate + MonthEnd(2)))

    offset = MonthEnd()
    print("offset = {}".format(offset))
    print("offset.rollforward(cdate) = {}".format(offset.rollforward(cdate)))
    print("offset.rollback(cdate) = {}".format(offset.rollback(cdate)))

    # 2 Methods to get the mean of every Month
    #   ts.groupby(MonthEnd().rollforward).mean(); ts.resample('M', how='mean').
    #
    K = 20
    ts = Series(np.random.randn(K),
                index = pd.date_range('1/15/2000', periods=K, freq='4d'))
    grouped = ts.groupby(offset.rollforward)
    print("ts.groupby(offset.rollforward).mean():\n{}".format(grouped.mean()))
    print("same result with resample('M', how='mean'):\n{}".format(ts.resample('M', how='mean')))

    # TS visualization
    fstock = os.path.join(DATA_PATH, r'stock_px.csv')
    close_px_all = pd.read_csv(fstock, parse_dates=True, index_col=0)
    close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
    print("close_px before resample:\n{}".format(close_px))
    close_px = close_px.resample('B').ffill()
    print("close_px after resample:\n{}".format(close_px))
    print("close_px.info():\n{}".format(close_px.info()))

    close_px['AAPL'].plot()
    close_px.ix['2009'].plot()
    close_px['AAPL'].ix['01-2011': '03-2011'].plot()
    print("close_px['AAPL'] before resample:\n{}".format(close_px['AAPL']))
    appl_q = close_px['AAPL'].resample('Q-DEC').ffill()
    print("close_px['AAPL'] after resample:\n{}".format(appl_q))
    appl_q.ix['2009':].plot()

    # different as close_px = close_px.asfreq('B', how='ffill') <= np.nan is in the result
    # same as close_px = close_px.asfreq('B', method='ffill')
    # same as close_px = close_px.asfreq('B').fillna(method='ffill')
    close_px = close_px.asfreq('B').fillna(method='ffill').ffill()
    print("close_px after asfreq('B').fillna(method='ffill').ffill():\n{}").format(close_px)

    close_px.AAPL.plot()
    plt.figure()
    plt.plot(close_px.AAPL.rolling(250).mean())
    close_px.AAPL.rolling(250).mean().plot()

    appl_std250 = close_px.AAPL.rolling(250, min_periods=10).std()
    print("appl_std250[5:12]:\n{}".format(appl_std250[5:12]))
    appl_std250.plot()

    expanding_mean = lambda x: x.rolling(len(x), min_periods=1).mean()

    close_px_m60 = close_px.rolling(60).mean()
    close_px_m60.plot(logy=True)

    plt.close('all')

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True,
                             figsize=(12, 7))
    aapl_px = close_px.AAPL['2005': '2009']
    ma60 = aapl_px.rolling(60, min_periods=60).mean()
    # here ewma can't use span=60 parameter but min_periods=60
    ewma60 = aapl_px.ewm(60, min_periods=60).mean()

    aapl_px.plot(style='k-', ax=axes[0])
    ma60.plot(style='k--', ax= axes[0])
    aapl_px.plot(style='k-', ax=axes[1])
    ewma60.plot(style='k--', ax=axes[1])
    axes[0].set_title('Simple MA')
    axes[1].set_title('Exponentially-weighted MA')

    print("close_px after resample('B', method='ffill'):\n{}".format(close_px))
    spx_px = close_px_all.SPX

    spx_rets = spx_px / spx_px.shift(1) - 1
    returns = close_px_all.pct_change()
    corr = returns.AAPL.rolling(125, min_periods=100).corr(spx_rets)
    plt.figure()
    corr.plot()
    corr = returns.rolling(125, min_periods=100).corr(spx_rets)
    corr.plot()

    # percentileofscore:
    #       The percentile rank of a score relative to a list of scores.
    #  A `percentileofscore` of, for example, 80% means that 80% of the scores in `a`
    #       are below the given score.
    #       In the case of gaps or ties, the exact definition depends on the optional keyword, `kind`.
    #  Parameters
    #  ----------
    #  a: array_like
    #      Array of scores to which `score` is compared.
    #  score : int or float
    #      Score that is compared to the elements in `a`.
    #  kind : {'rank', 'weak', 'strict', 'mean'}, optional
    #
    from scipy.stats import percentileofscore
    score_at_2percent = lambda x: percentileofscore(x, 0.02)
    result = returns.AAPL.rolling(250).apply(score_at_2percent)
    plt.figure()
    result.plot()
    plt.close('all')

    # 时序序列分析 TSA
    # initialize the parameter
    fdisc = os.path.join(DATA_PATH, r'arima_data.xls')
    forecastnum = 5

    # read in data:
    #   当“日期”格式为默认格式 %Y-%M-%D %H:%M:%S时，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
    #   如果是其他格式，可以用 date_parser 指定处理函数。
    #
    data = pd.read_excel(fdisc, index_col=u'日期')
    print("data read in from {}:\n{}".format(fdisc, data))
    print("\ndata.info():\n{}".format(data.info()))
    print("\ndata.describe():\n{}".format(data.describe()))
    data = pd.DataFrame(data, dtype=np.float64)
    print("data after converted to DataFrame(...dtype=np.float64):\n{}".format(data))
    # 时序图
    data.plot()
    plt.show()

    # 自相关图
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plot_acf(data).show()

    # 平稳性检测
    # adfuller = Augmented Dickey-Fuller unit root test
    #   The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root,
    #       with the alternative that there is no unit root.
    #   If the pvalue is above a critical size, then we cannot reject that there is a unit root.
    #   ADF 返回值为 adf, pvalue, usedlag, nobs, critical values, icbest, resstore.
    #       nobs = number of obsovations,
    #       icbest = max information criterion if autolag is not None, autolag=AIC by default.
    #
    from statsmodels.tsa.stattools import adfuller as ADF
    print(u"平稳性检测 ADF 的结果：\n{}".format(ADF(data[u'销量'])))

    # 差分后结果
    Ddata = data.diff().dropna()
    Ddata.columns = [u'销量差分']
    Ddata.plot()
    plt.show()
    plot_acf(Ddata).show()
    plot_pacf(Ddata).show()
    print(u"差分后，平稳性检测 ADF 的结果：\n{}".format(ADF(Ddata[u'销量差分'])))

    # 白噪声检验
    #   acorr_ljungbox: Ljung-Box test for no autocorrelation
    #       Ljung-Box and Box-Pierce statistic differ in their scaling of the autocorrelation function.
    #       Ljung-Box test is reported to have better small sample properties.
    #   acorr_ljungbox 返回值：统计量和p-value
    #   如果 p-value 远小于 0.05,则序列是非白噪声。
    #   ref:    https://zhuanlan.zhihu.com/p/35128342
    #
    from statsmodels.stats.diagnostic import acorr_ljungbox
    print(u"白噪声检验 acorr_ljungbox 结果：\n{}".format(acorr_ljungbox(Ddata, lags=1)))

    # ARIMA 模型
    from statsmodels.tsa.arima_model import ARIMA
    # 定阶， 阶数一般不超过length/10.
    pmax = int(len(Ddata) / 10)
    qmax = int(len(Ddata) / 10)
    bic_matrix = []
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
            except Exception as e:
                tmp.append(None)
        bic_matrix.append(tmp)

    print("bic_matrix original values:\n{}".format(bic_matrix))
    # 为了找出 p,q 最小值位置
    bic_matrix = pd.DataFrame(bic_matrix)
    print("bic_matrix after converted to DataFrame:\n{}".format(bic_matrix))
    p, q = bic_matrix.stack().idxmin()
    print(u"BIC最小的p值 和 q值：p={}\tq={}\t".format(p, q))

    model = ARIMA(data, (p, 1, q)).fit()
    print("ARIMA model created with min_p_and_q as (p, 1, q):\n{}".format(model))
    # to switch defaultencoding from ascii to utf-8
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    # model.summary() 给出一份模型报告
    print(u"model.summary():\n{}".format(model.summary()))
    # model.forecast(5) 作为期5天的预测，返回预测结果、标准误差、置信区间。
    print("model.forecast({}):\n{}".format(forecastnum, model.forecast(forecastnum)))

# classes

# main entry
if __name__ == "__main__":
    main()
    