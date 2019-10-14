#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        numpy_practice.py

    DESCRIPTION
        to practice numpy in Python
        1）均匀分布： np.random.randint(参数为low,high,size）,  .rand（简便方法，参数为维度）,
                   .random == .random_sample （返回0-1之间的float）,.uniform(参数为low,high,size）;
        正态分布： .randn（简便方法，参数为维度）, .normal, .standard_normal（mean=0, sigma=1).
        随机排列： .permutation（一个随机排列）

        uniform(low=0.0, high=1.0, size=None) : Draw samples from a uniform distribution.
                randint : Discrete uniform distribution, yielding integers.
        random_integers : Discrete uniform distribution over the closed
                          interval ``[low, high]``.
        random_sample : Floats uniformly distributed over ``[0, 1)``.
        random : Alias for `random_sample`.
        rand : Convenience function that accepts dimensions as input, e.g.,
               ``rand(2,2)`` would generate a 2-by-2 array of floats,

        orig_data = np.random.randint(0, 2, size=STEPS_MAX)

        2） np.msort(), np.where() + np.take(), np.average() vs np.mean(), np.ravel(np.where()),
            np.split(), np.maximum() vs np.max(),
            np.fill(),
            np.apply_along_axis(...),
            np.convolve(weight, signal, 'full'_or_'same'_or_'valid'): 卷积实质上是加权平均积，卷积是频域上的乘积！
                        所以，时域的卷积，也可以用频域的乘积计算。
                        对时不变信号：给产品一个脉冲信号，能量是1焦耳，输出的波形图画出来！
                                    对于某个输入波形，你想象把它微分成无数个小的脉冲，输入给产品，
                                    叠加出来的结果就是你的输出波形。
                                    你可以想象这些小脉冲排着队进入你的产品，每个产生一个小的输出，
                                    你画出时序图的时候，输入信号的波形好像是反过来进入系统的。
                        卷积是分析数学中一种重要的运算，定义为一个函数与经过翻转和平移的另一个函数的乘积的积分。
                        weights不变，signal翻转或者旋转180度；signal从左到右移动，与weights乘积相加，得到卷积结果。
                        图像的卷积有三种模式：full、same和valid本章就这三种模式进行讨论(这里假设移动步长stride = 1)。
                                            这三种模式实际是在核k对输入数据i滑动卷积过程中有重叠但有的时候不是全部重叠在一起，
                                            核k可能有一部分和输入数据重合而另一部分漂在输入数据外部，
                                            技术上对漂在外部的处理方式是填零Zero padding。
                        由于乘法比卷积更有效（更快），函数scipy.signal.fftconvolve利用FFT来计算大数据集的卷积。
            np.linalg.lstsq(), np.linalg.solve()

            c_cp_sorted = np.msort(c_cp)
            indices = np.where(wday == i)
            prices = np.take(c, indices)
            last_friday = np.ravel(np.where(wday == 4))[3]
            weeks_indices = np.split(weeks_indices, 3)
            averages = np.zeros(N)
            averages.fill(sma[i-N+1])
            weeksummary = np.apply_along_axis(summarize, 1, weeks_indices, o, h, l, c)

            sma = np.convolve(weights, c)[N-1:-N+1]
            #coding:utf-8
            import numpy as np
            from scipy import fftpack,signal
            import matplotlib.pyplot as plt
            x = np.array([[1.0, 1.0, 0.0],[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
            h = np.array([[1.0, 0.0], [0.0, 1.0]])
            y = signal.convolve(x, h, mode = "full")
            print y

            (x, residuals, rank, s) = np.linalg.lstsq(A, b)
            Adotx = np.dot(A, x)
            print('Double check if (A np.dot x) ~=b:\n{}\n\tTheir difference:\n\t{}'.format(
                Adotx, ['{:1.13f}'.format(i) for i in (Adotx - b)]))
            print('np.allclose(A.dot(x), b)? {}'.format(np.allclose(A.dot(x), b)))  // True

            sx = np.linalg.solve(A, b)
            print('sx == x? {}'.format(sx == x))        // False
            print('np.allclose(sx, x)? {}'.format(np.allclose(sx, x))) // True


        # ref: http://www.voidcn.com/article/p-htlgvvsr-bkz.html
                http://www.voidcn.com/article/p-pzsezzko-xs.html
                http://www.voidcn.com/article/p-umovxskn-bpt.html
                https://www.osgeo.cn/numpy/reference/generated/numpy.convolve.html
                http://liao.cpython.org/scipytutorial19/
                http://liao.cpython.org/scipytutorial20/
                https://www.cnblogs.com/21207-iHome/p/6231607.html
                http://doc.codingdict.com/NumPy_v111/reference/generated/numpy.convolve.html




    MODIFIED  (MM/DD/YY)
        Na  11/18/2018

"""
__VERSION__ = "1.0.0.11182018"


# imports
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt

# consts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')

# functions
def datestr2num(date=None):
    return dt.datetime.strptime(date, '%d-%m-%Y').weekday()

def datestr2dt(date=None):
    return np.datetime64('{2}-{1}-{0}'.format(*date.split('-')))

def main():
    # Part 1
    # np.random
    STEPS_MAX = 1000
    orig_data = np.random.randint(0, 2, size=STEPS_MAX)
    steps = np.where(orig_data > 0, 1, -1)
    walks = steps.cumsum()
    K = 15
    steps_overk = walks[walks > K]
    print('Walks with {} randomized step:\n{}\nMax value: {}\nMin value: {}'.format(
        STEPS_MAX, walks, walks.max(), walks.min()))
    print('Totally {} steps walked over {}:\n{} '.format(steps_overk.size, K, steps_overk))
    print('The max step used to walk max {} step: {}'.format(
        walks.max(), walks.argmax()))

    # Part 2
    # to analyze stock price
    # content format:
    #   name,date,NA,open,high,low,close,volume
    #   AAPL,28-01-2011, ,344.17,344.4,333.53,336.1,21144800
    f1 = os.path.join(DATA_DIR, 'data.csv')
    print('f1: {}'.format(f1))

    # hard code, how to read data from a text file in specific dtype: use dtype option
    name = 'AAPL'
    wday,o, h, l, c, v = np.loadtxt(f1, delimiter=',', usecols=(1,3,4,5,6,7),
                                            converters={1: datestr2num}, unpack=True)
    # VWAP, TWAP
    vwap = np.average(c, weights=v)
    twap = np.average(c, weights=np.arange(len(c)))
    print('Analysis for historical stock price of {}:'.format(name))
    print('Original data of close price:\n{}'.format(c))
    print('VWAP: {}\nTWAP: {}\nMean: {}'.format(vwap, twap, c.mean()))
    print('VWAP: {}\nTWAP: {}\nMean: {}'.format(vwap, twap, c.mean()))

    # max, min, ptp
    highest, lowest = h.max(), l.min()
    print('Highest price: {}\nLowest  price: {}\nMean of highest and lowest price: {}'.format(
        highest, lowest, (highest + lowest) / 2 ))
    print('High price spread over: {}\nLow  price spread over: {}'.format(
        np.ptp(h), np.ptp(l)))

    # statistics
    c_cp = np.copy(c)
    c_cp_sorted = np.msort(c_cp)
    print('Median: {}\nSorted:\n{}'.format(np.median(c_cp), c_cp_sorted))
    clen = len(c_cp)
    print('Middle: {}\nAvearage middle: {}'.format(
        c_cp_sorted[(clen -1)/2], (c_cp_sorted[clen/2] + c_cp_sorted[(clen-1)/2]) / 2 ))
    print('Variance: {}\nVariance from definition: {}'.format(
        np.var(c), np.mean((c - c.mean())**2) ))

    # stock profit rate
    returns = np.diff(c) / c[:-1]
    print('Returns: \n{}'.format(returns))
    print('\tMean: {}\n\tStandard deviation: {}\n\tVariance: {}'.format(
        returns.mean(), returns.std(), returns.var() ))
    logreturns = np.diff( np.log(c) )
    print('Returns in log format: \n{}'.format(logreturns))
    print('\tMean:{}\n\tStandard deviation: {}\n\tVariance: {}'.format(
        np.mean(logreturns), np.std(logreturns), np.var(logreturns)))

    posretindices = np.where(returns > 0)
    print('Indices with positive returns: {}\n'.format(posretindices))

    annual_volatility = logreturns.std() / logreturns.mean() / np.sqrt(1./252.)
    print('Annual  Volatility: {}'.format(annual_volatility))
    print('Monthly Volatility: {}'.format(annual_volatility * np.sqrt(1./12.)))

    # analysis based on date
    print('Dates:\n{}'.format(wday))
    averages = np.empty(5)
    for i in xrange(5):
        indices = np.where(wday == i)
        prices = np.take(c, indices)
        avg = prices.mean()
        print('Day {} prices {} Average {}'.format(i, prices, avg))
        averages[i] = avg

    print('Highest average: {},\tTop     day of the week: {}'.format(averages.max(), averages.argmax()))
    print('Lowest  average: {},\tBotteom day of the week: {}'.format(averages.min(), averages.argmin()))

    # weekly summary, using data of 3 weeks as an example, i.e. weekday: 0-4 0-4 0-4
    first_monday = np.ravel(np.where(wday == 0))[0]
    last_friday = np.ravel(np.where(wday == 4))[3]
    print('The first Monday index is {}.\nThe last Friday index is {}'.format(first_monday, last_friday))

    weeks_indices = np.arange(first_monday, last_friday+1)
    print('Weeks indices initial: {}'.format(weeks_indices))
    weeks_indices = np.split(weeks_indices, 3)
    print('Weeks indices after split: {}'.format(weeks_indices))

    def summarize(w_idx, o, h, l, c):
        monday_open = o[w_idx[0]]
        week_high = np.take(h, w_idx).max()
        week_low = np.take(l, w_idx).min()
        friday_close = c[w_idx[-1]]
        return(name, monday_open, week_high, week_low, friday_close)

    weeksummary = np.apply_along_axis(summarize, 1, weeks_indices, o, h, l, c)
    print('Week summary:\n{}'.format(weeksummary))

    np.savetxt(os.path.join(DATA_DIR, 'weeksummary.csv'), weeksummary, delimiter=',', fmt='%s')

    # ATR 真实波动幅度均值
    N = 20
    h_atr, l_atr = h[-N:], l[-N:]
    print('len(h): {} \t len(l): {}'.format(len(h_atr), len(l_atr)))
    print('len(c): {}\t Close:\n{}'.format(len(c), c))
    previousclose = c[-N-1:-1]
    print('len(previousclose): {}\tPrevious close:\n{}'.format(len(previousclose), previousclose))
    truerange = np.maximum(h_atr - l_atr, h_atr - previousclose, previousclose - l_atr)
    print('True range:\n{}'.format(truerange))

    atr = np.zeros(N)
    atr[0] = truerange.mean()
    for i in xrange(1, N):
        atr[i] = ((N - 1) * atr[i-1] + truerange[i] ) / N
    print('ATR:\n{}'.format(atr))

    # SMA简单移动平均线, EMA指数移动平均线, BolingerBand布林带 # taking 5 days as average
    N = 5
    weights = np.ones(N) / N
    print('Weights for SMA: {}'.format(weights))
    sma = np.convolve(weights, c)[N-1:-N+1]
    t = np.arange(N-1, len(c))
    plt.plot(t, c[N-1:], lw=1.0)
    plt.plot(t, sma, lw=2.0)
    plt.title('SMA')
    plt.show()

    # EMA
    weights = np.exp(np.linspace(-1., 0., N))
    weights /= weights.sum()
    print('Weights for EMA: {}'.format(weights))
    ema = np.convolve(weights, c)[N-1:-N+1]
    t = np.arange(N-1, len(c))
    plt.plot(t, c[N-1:], lw=1.0)
    plt.plot(t, ema, lw=2.0)
    plt.title('EMA')
    plt.show()

    # BolingerBand
    deviation = []
    averages = np.zeros(N)
    for i in xrange(N-1, clen):
        dev = c[i: i+N] if ((i + N) < clen) else c[-N:]
        averages.fill(sma[i-N+1])
        dev = (dev - averages) ** 2
        dev = np.sqrt(dev.mean())
        deviation.append(dev)

    deviation = 2 * np.array(deviation)
    print('len(deviation): {}\t len(sma): {}'.format(len(deviation), len(sma)))

    upperBB, lowerBB = sma + deviation, sma - deviation
    c_slice = c[N-1:]
    between_bands = np.where((c_slice < upperBB) & (c_slice > lowerBB))
    # print('[TO debug only] between_bands = {}\n\t\ttype(between_bands) = {}'.format(
    #     between_bands, type(between_bands)))

    print('lowerBB[between_bands]: \n{}\nc[between_bands]: \n{}\nupperBB[between_bands]: \n{}'.format(
        lowerBB[between_bands], c[between_bands], upperBB[between_bands] ))
    print('Ratio between bands: {:1.4f}'.format(float( len(np.ravel(between_bands)) / len(c_slice))))

    plt.plot(t, c_slice, lw=1.0)
    plt.plot(t, sma, lw=2.0)
    plt.plot(t, upperBB, lw=3.0)
    plt.plot(t, lowerBB, lw=4.0)
    plt.title('Bolinger Band')
    plt.show()

    # linear model
    # N = int(sys.argv[1])
    N = int(input('Please input N (0 < an integer < 30 ) used to fit in a linear model: '))
    c = np.loadtxt(f1, delimiter=',', usecols=(6,), unpack=True)

    b = c[-N:][::-1]
    print('N = {}\nc:\n{}\nb:\n{}'.format(N, c, b))

    A = np.zeros((N, N), float) # by default it's float64.
    print('A = Zeros N by N:\n{}'.format(A))

    for i in xrange(N):
        A[i, ] = c[-N-1-i: -1-i]
    print('A after filled with data from c:\n{}'.format(A))

    (x, residuals, rank, s) = np.linalg.lstsq(A, b)
    print('Used np.linalg.lstsq to simulate:\nx = {}\nresiduals = {}\nrank = {}\ns = {}'.format(
        x, residuals, rank, s))
    Adotx = np.dot(A, x)
    print('Double check if (A np.dot x) ~=b:\n{}\n\tTheir difference:\n\t{}'.format(
        Adotx, ['{:1.13f}'.format(i) for i in (Adotx - b)]))
    print('np.allclose(A.dot(x), b)? {}'.format(np.allclose(A.dot(x), b)))
    sx = np.linalg.solve(A, b)
    print('from np.linalg.solve(A, b):\n\t{}'.format(sx))
    print('sx == x? {}'.format(sx == x))
    print('np.allclose(sx, x)? {}'.format(np.allclose(sx, x)))

    ## following part could be run separately
    import numpy as np
    import matplotlib.pyplot as plt
    f1 = r'D:/workspace/dataguru/pda/data/data.csv'

    # trend line
    def fit_line(t, y):
        A = np.vstack([t, np.ones_like(t)]).T
        lstsq_result = np.linalg.lstsq(A, y)
        # print('[DEBUG] lstsq_result = \033[1;33m{}\033[0m'.format(lstsq_result))
        return lstsq_result[0]

    h, l, c = np.loadtxt(f1, delimiter=',', usecols=(4, 5, 6), unpack=True)
    pivots = (h + l + c) / 3
    print('Pivots:\n\t{}'.format(pivots))

    len_c = len(c)
    t = np.arange(len_c)
    sa, sb = fit_line(t, pivots - (h - l))
    ra, rb = fit_line(t, pivots + (h - l))

    support = sa * t + sb
    resistance = ra * t + rb
    condition = (support < c) & (c < resistance)
    print('Condition:\n{}'.format(condition))
    between_sr_bands = np.where(condition)
    print('support[between_sr_bands]:\n\t{}\nc[between_sr_bands]:\n\t{}\nresistance[between_sr_bands]:\n\t{}'.format(
        support[between_sr_bands], c[between_sr_bands], resistance[between_sr_bands]))
    len_bbands = len(np.ravel(between_sr_bands))
    print('Number of points between support and resistance: {}'.format(len_bbands))
    print('Ratio of points between support and resistance: {}'.format(float(len_bbands) / len_c))

    print("Tomorrow's support   : {}".format(sa * (t[-1] + 1) + sb))
    print("Tomorrow's resistance: {}".format(ra * (t[-1] + 1) + rb))

    a1, a2 = c[c > support], c[c < resistance]
    print('Number of points between support and resistance 2nd approach: {}'.format(len(np.intersect1d(a1, a2))))

    plt.plot(t, c)
    plt.plot(t, support)
    plt.plot(t, resistance)
    plt.show()




# classes

# main entry
if __name__ == "__main__":
    main()
    