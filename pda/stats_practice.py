#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        stats_practice.py

    DESCRIPTION
        1. 统计学可以分为描述统计学或推断统计学：
            描述统计学：使用特定的数字或图表来体现数据的集中程度和离散程度，
                        例如：每次考试的平均分、最高分、各个分段的人数分布等，也是属于描述统计学的范围。
            <=> 集中程度、离散程度、分布形态（偏度、峰值）
                均值u:        np.mean,a.mean, s.mean, df.mean(axis=...)
                              代表平均水平；充分利用所有数据，适应性强，但容易受到极端值影响；
                中位数median:  np.median, s.median, df.median(axis=...)
                               代表中间水平；不受极端值影响，但是缺乏敏感性；
                众数mode:      collections.Counter(a).most_common(1),
                                s.mode(), df.mode(axis=..., numeric_only=...)
                               代表一般水平；不受极端值影响；当数据具有明显的集中趋势时，代表性好；
                               但是缺乏唯一性，可能有一个，可能有两个，可能一个都没有。
                极差ptp:        a.ptp(), s.ptp(), np.ptp(df, axis=...), np.ptp(df.data1)
                数据分析中方差分样本方差，总体方差，标准差类似：
                方差：         a.var(), s.var(), df.var(axis=...)，有样本方差S²，总体方差σ²;
                                                                 1
                               总体方差用σ²表示，计算公式：σ²= ─  Σᵢⁿ₌ ₁ (Xᵢ−𝜇)²；
                                                                𝑁
                                                                  1
                               样本方差用 S²表示，计算公式： S²=  ─  Σᵢⁿ₌ ₁ (Xᵢ−𝜇)²。
                                                                𝑁-1


                标准差：        a.std(), s.std(), df.std(axis=...)，是方差的平方根，相应的有样本标准差S，总体标准差σ。

                偏度skewness:   nan, s.skew(), df.skew(axis=...)  # if df.shape==(M, N), M, N >=3 otherwise skew=nan
                                对数据分布的偏斜程度的衡量。
                                正偏： >0, 负偏： <0.

                峰值kurtosis:   nan, s.kurt(), df.kurt(axis=...)  # if df.shape==(M, N), M, N >=4 otherwise kurt=nan
                                数据分布峰态的度量指标；
                                尖峰、中锋、低峰，超额峰度。

                分位数：        np.quantile(a, [0.01, 0.25, 0.75, 1.]), np.quantile(a, 0.25)，
                                s1.quantile(), s1.quantile([0.01, 0.25, 0.75, 1.])，
                                df.quantile(), df.quantile([0.01, 0.25, 0.75, 1.])
                                将数据从小到大的顺序分为两组，较小的一组的元素个数占整个样本元素个数的。
                                上四分位数，下四分位数。
                可以用 s1.describe(), df.describe() 来描述数据的基本特征。

                协方差cov：     nan, s.cov(s2 or ...), df.cov()
                                df.cov:
                                    Compute pairwise covariance of columns, excluding NA/null values.
                                    Compute the pairwise covariance among the series of a DataFrame.
                                    The returned data frame is the `covariance matrix` of the columns of the DataFrame.
                                    This method is generally used for the analysis of time series data to
                                    understand the relationship between different measures across time.
                                用 Sxx 表示变量 X 观测样本的方差，Syy 表示变量 Y 观测样本的方差，
                                用 Sxy 表示变量 X，Y 的观测样本的协方差，称
                                    S = Sxx Sxy
                                        Sxy Syy
                                为观测样本的协方差矩阵，
                                称 r=Sxy/(.sqrt(Sxx) * .sqrt(Syy)) 为观测样本的相关系数。
                                    Sxy = .sum((x_i - x_bar) * (y_i - y_bar)) / (n-1)
                                协方差表示两个变量的总体的变化趋势。
                                    如果两个变量的变化趋势一致，也就是说如果其中一个大于自身的期望值，另外一个也大于自身的
                                        期望值，那么两个变量之间的协方差就是正值。
                                    如果两个变量的变化趋势相反，即其中一个大于自身的期望值，另外一个却小于自身的期望值，
                                        那么两个变量之间的协方差就是负值。
                                    如果两个变量不相关，则协方差为 0，变量线性无关不表示一定没有其他相关性。

                                    作者：wyrover
                                    链接：https://www.jianshu.com/p/738f6092ef53
                                    來源：简书
                                    简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。

                相关系数corr：   nan, s.corr(s2 or ...), df.corr(), df.corrwith(df2 or ...)

            推断统计学：根据样本数据推断总体数据特征。
                        例如：产品质量检查，一般采用抽检，根据所抽样本的质量合格率作为总体的质量合格率的一个估计。
            <=> 假设检验
                    基本原理：
                        小概率思想
                        反证法思想

                    基本概念：
                        零假设与备选假设——无罪推定原理
                            零假设 null hypothesis：假定一个总体参数等于某个特定值的一个声明，用 H₀ 表示。
                                    若希望假设的论断不成立，设为零假设。
                            null hypothesis is a statement that the value of a population parameter
                                is equal to some claimed value.
                                如： H₀: p=0.5; H₀: u=98.6; H₀: σ=15
                            备择假设： 假定该总体参数为零假设中假设的值除外的值，用 H₁ 表示。
                                如： H₁: p>0.5; H₁: p<0.5; H₁: pǂ0.5

                        检验统计量 test statistic, t: 检验统计量是一个用于确定零假设是否为真的一个值，
                                                            这个值在假定零假设为真时由样本数据计算得到的。
                                                    有z检验、t检验、卡方检验、F检验。
                        The test statistic is a value used in making a decision about the null hypothesis,
                            and it is found by converting the sample statistic to a score with the assumption
                                that the null hypothesis is true.
                        t 可以用 scipy.stats 计算，也可以手动计算。
                            se = data.std() / np.sqrt(data.size)   # 计算标准误差： 样本标准差 / （n的开方）
                            t = (data.mean() - popmean) / se
                        在线计算p-value： 将t值和自由度v=n-1代入 Statistical distributions and interpreting P values
                                          http://link.zhihu.com/?target=https%3A//www.graphpad.com/quickcalcs/distMenu/
                                        中可得双尾t检验的p值为...。

                        拒绝域： 也称否定域，是指检验统计量所有可以拒绝零假设的取值所构成的集合。
                                    计算得出的t_statistic > t临界值t_ci 就是拒绝域。

                        显著性水平 α：指当零假设正确的时候，检验统计量落在拒绝域的概率。
                                    也就是当零假设为真而我们却拒绝零假设这种错误发生的概率。
                                    与置信区间中的显著性水平意义一致。
                                    常用取值：0.1, 0.05, 0.01.
                            \alpha 在 plt 中画出 alpha，也可以用 sympy.pprint(sympy.abc.alpha) 输出 α。

                        临界值 t_ci：拒绝域与非拒绝域的分界线。可以用 scipy.stats 计算，也可以手动计算。

                        根据自由度n-1和α查找t临界值表，计算1-α=95% 的置信水平

                        效应量：表示量化显著差异。
                                Cohen's d = (data.mean() - popmean) / data.std()
                                查效应量Cohen's d绝对值和效果显著（差异大小）含义的对应表，可得知效果是否显著（差异大小）。

                        # https://www.cnblogs.com/emanlee/archive/2008/10/25/1319520.html
                        # ref: https://zhuanlan.zhihu.com/p/29284854
                        # ref: https://zhuanlan.zhihu.com/p/36727517


                        决定规则：3条：
                        \-------------\----------------------------\----------------------------------
                        \    方法      \       拒绝零假设           \   不拒绝零假设
                        \ 临界值法；   \     检验统计量落在拒绝域；  \   检验统计量没有落在拒绝域；
                        \ P-value法；  \     P-value<=\alpha；      \   P-value>\alpha；
                        \另一个选择 ； \    不采用具体的 \alpha 值，写出 p-value 留给读者自己判断。
                        \--------------\--------------------------------------------------------------

                        两类错误：第一类错误：零假设为真，拒绝零假设；
                                  第二类错误：零假设为假，不拒绝零假设。

                    基本概率：
                        p-value: 样本发生或者比样本更极端的情况发生的概率。

                    假设检验的基本步骤：
                        1） 提出零假设；
                        2） 建立检验统计量；
                        3） 确定否定域/计算p-value;
                        4） 得出结论。

        2.  To print out math symbols via sympy
            pip install sympy

            sympy:      SymPy is a Python library for symbolic mathematics.
                        It aims to become a full-featured computer algebra system (CAS)
                            while keeping the code as simple as possible
                            in order to be comprehensible and easily extensible.
            sympy.abc: This module exports all latin and greek letters as Symbols

            from sympy import abc
            from sympy import *

            # pretty print latin and greek letters
            >>> pprint(sympy.abc.alpha)
            α

            >>> pretty(sympy.abc.sigma)
            u'\u03c3'
            >>> print(u'\u03c3')
            σ
            pprint(sympy.abc.mu)
            pprint(sympy.abc.sigma)
            ...

            >>> print(u'I am {}'.format(pretty(pi)))
            I am π
            >>> print(u'\u03c0')
            π

            >>> sympy.pprint(sympy.abc.epsilon)
            ε

            >>> import unicodedata
            >>> print(u'I am {}'.format(unicodedata.lookup('GREEK SMALL LETTER ALPHA')))
            I am α
            >>> print(unicodedata.lookup('GREEK CAPITAL LETTER SIGMA'))
            Σ

            >>> print(unicodedata.lookup('latin small letter y with circumflex'))
            ŷ

            # subscript
            >>> pprint(symbols('H_0'))
            H₀
            >>> pprint(symbols('H_1'))
            H₁

            # supscript
            >>> pprint(symbols(u'{}^2'.format(pretty(sympy.abc.sigma))))
            σ²



            # ref: http://www.asmeurer.com/sympy_doc/dev-py3k/tutorial/tutorial.zh.html
            # ref: https://www.cnblogs.com/sdlypyzq/p/5382755.html
            # ref: https://blog.csdn.net/mandagod/article/details/64905549
            # ref: https://stackoverflow.com/questions/26483891/printing-greek-letters-using-sympy-in-text
            # ref: https://stackoverflow.com/questions/24897931/how-do-i-define-a-sympy-symbol-with-a-subscript-string

            # math symbols
            import unicodedata

            >>> print(ord('='))
            61
            >>> print(chr(61))
            =
            >>> print('{}{}'.format(chr(33), chr(61)))
            !=

            >>> print(unicodedata.lookup('LATIN LETTER ALVEOLAR CLICK'))
            ǂ

            >>> print(u'\u2248')
            ≈
            >>> print(unicodedata.lookup('ALMOST EQUAL TO'))
            ≈

            # ref:  https://www.dcl.hpi.uni-potsdam.de/home/loewis/table-3131.html
            # ref:  https://books.google.co.in/books?id=A0wOCgAAQBAJ&pg=PA15&lpg=PA15&dq=python+%E6%89%93%E5%8D%B0+%E2%89%88&source=bl&ots=4PQXz885Gx&sig=p60087ed05O7QU5lfcCx-4BEFk4&hl=en&sa=X&ved=2ahUKEwjhjb63vr3fAhUOfFAKHaptCicQ6AEwA3oECAUQAQ#v=onepage&q=python%20%E6%89%93%E5%8D%B0%20%E2%89%88&f=false
            # ref:  常用数学符号大全 https://blog.csdn.net/mutex86/article/details/9138947


        3. 数据分布
            二项分布就是重复n次独立的伯努利试验。在每次试验中只有两种可能的结果，而且两种结果发生与否互相对立，并且相互独立，
                    与其它各次试验结果无关，事件发生与否的概率在每一次独立试验中都保持不变，
                    则这一系列试验总称为n重伯努利实验。
            二项分布是n个独立的是/非试验中成功的次数的离散概率分布，其中每次试验的成功概率为p。
                    这样的单次成功/失败试验又称为伯努利试验。实际上，当 n=1 时，二项分布就是伯努利分布。

                    计算概率的一般公式： b(x,n,p) = Cnˣ pˣ qⁿ-ˣ，
                                        其中，b表示二项分布的概率，n试验次数，x事件A发生的次数，p事件A发生的概率
                                              Cnˣ 是组合，表示在 n 次实验中出现 x 次结果的可能的次数。
                    二项分布频繁地用于对以下描述的一种实验进行建模：从总数量大小为N的两个事物中进行n次放回抽样，
                            以某一事物为基准，计算成功抽取这个事物的次数的概率。
                            要注意的是必须进行的是放回抽样，对于不放回抽样我们一般用超几何分布来对这样的实验进行建模。

                    # ref:  https://baike.baidu.com/item/%E4%BA%8C%E9%A1%B9%E5%88%86%E5%B8%83
                    # ref:  https://blog.csdn.net/huangjx36/article/details/77990392
                    # ref:  https://zhuanlan.zhihu.com/p/24692791



    MODIFIED  (MM/DD/YY)
        Na  12/17/2018

"""
__VERSION__ = "1.0.0.12172018"


# imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import os
import pprint as pp

# configuration
np.random.seed(12345)
np.set_printoptions(precision=4, threshold=500)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format
plt.rc('figure', figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week9_data')
# CURRENT_PATH, DATA PATH here is for running code snippet in Python Console by Shift+Alt+E
# CURRENT_PATH = os.getcwd()
# DATA_PATH = os.path.join(CURRENT_PATH, r'pda\data\week9_data')

# functions
def main():
    a = [98, 83, 65, 72, 79, 76, 75, 94, 91, 77, 63, 83, 89, 69, 64, 78, 63, 86, 91, 72, 71, 72, 70, 80, 65, 70, 62, 74,
         71, 76]
    print("type(a): {}".format(type(a)))
    print("mean(a):\t{:,.4f}".format(np.mean(a)))
    print("median via mean(np.sort(a)[14:16]):\t{:,.4f}, via np.median(a):\t{:,.4f}".
          format(np.mean(np.sort(a)[14:16]), np.median(a)))
    print("np.sort(a):\n{}".format(np.sort(a)))

    # to calculate mode
    def get_mode(arr):
        mode = None
        arr_appear = dict((a, arr.count(a)) for a in arr)
        max_count = max(arr_appear.values())
        if max_count == 1:
            # No mode
            return None;
        else:
            # get mode
            # mode = []
            # for k, v in arr_appear.iteritems():
            #     if v == max_count:
            #         mode.append(k)
            mode = [k for k, v in arr_appear.iteritems() if v == max_count]
        return mode

    print("get_mode(a):\t{}".format(get_mode(a)))

    print("np.var(a):\t{:,.4f}".format(np.var(a)))
    print("np.std(a)\t{:,.4f}\n".format(np.std(a)))

    s1 = Series(a)
    print("type(s1):\t{}".format(type(s1)))
    print("s1.skew():\t{:,.4f}".format(s1.skew()))
    print("s1.kurt():\t{:,.4f}".format(s1.kurt()))
    print("s1.describe():\n{}\n".format(s1.describe()))

    N = 5
    df = DataFrame({'data1': np.random.randn(N),
                    'data2': np.random.randn(N)})
    print("type(df):\t{}".format(type(df)))
    print("df.cov():\n{}\n".format(df.cov()))
    print("df.corr():\n{}\n".format(df.corr()))
    print("\tsame corrcoef:\t{:,.4f}".format(df.data1.corr(df.data2)))

    # hypothetical test 假设检验
    # 例子：
    #   一件物品的重量，将其称了10次，得到的重量为10.1,10,9.8,10.5,9.7,10.1,9.9,10.2,10.3,9.9，
    #   假设所称出的物体重量服从正态分布，我们现在想知道该物品的重量是否显著不为10？

    # 1） 零假设：物体重量显著=10；备择假设：物体重量显著不为10
    # 2） 检验统计量：均值           <= here df.mean() = 10.0500
    # 3） 计算检验统计量 和 p-value  <= statistic=array([0.6547]), pvalue=array([0.5291])
    # 4） 结论：p-value > \alpha (here taking \alpha as 0.05) => 零假设成立，物体重量显著=10。

    from scipy import stats as ss

    ## for t-distribution
    # Example1
    df = DataFrame({'data': [10.1,10,9.8,10.5,9.7,10.1,9.9,10.2,10.3,9.9]})
    HN = 10
    print("Data: df:\n{}\n".format(df))
    print("df.mean():\n{}\n".format(df.mean()))
    print("ss.ttest_1samp(a=df, popmean={}):\n\033[1;31m{}\033[0m\n".
          format(HN, ss.ttest_1samp(a=df, popmean=HN)))
    # output:
    #   Ttest_1sampResult(statistic=array([0.6547]), pvalue=array([0.5291]))

    # Example2
    # 2. 某学生随机抽取了10包一样的糖并称量它们的包装的重量，判断这些糖的包装的平均重量是否为3.5g。
    # 其中，这10包糖的重量如下（单位：g）：
    #     3.2,3.3,3.0,3.7,3.5,4.0,3.2,4.1,2.9,3.3
    from scipy import stats as ss
    df = DataFrame({'data': [3.2,3.3,3.0,3.7,3.5,4.0,3.2,4.1,2.9,3.3]})

    # check distribution shape, if similar to normal distribution
    plt.ion()
    df.plot(kind='hist', title=u'数据集分布')
    df.plot(kind='kde', ax=plt.gca())
    plt.pause(5)
    print(u'数据集来自于N次伯努利试验，分布类似正态分布 => \033[1;32m符合t分布，用t检验处理。\033[0m')
    plt.close()

    popmean = 3.5
    print(u'问题：某学生随机抽取了10包一样的糖并称量它们的包装的重量，判断这些糖的包装的平均重量是否为3.5g。')
    print(u'解答：\n1. 设定原假设：这些糖的包装的平均重量等于3.5g\n\t 备择假设：这些糖的包装的平均重量不等于3.5g')
    print(u'2. 设定检验统计值: 这些糖的平均重量 = {}'.format(popmean))
    t_statistic, p_value = ss.ttest_1samp(a=df, popmean=popmean)
    print(u'3. 计算得出：statistic = {}，p-value = {}'.format(t_statistic, p_value))

    # calculate t manually
    ddata = df.data
    dmean = ddata.mean()
    # 计算标准误差： 样本标准差 / （n的开方）
    se = ddata.std() / np.sqrt(ddata.size)
    # # 用 ss.sem() 计算
    # se2 = ss.sem(ddata)
    # print(u'手动计算的标准误差 == 用ss.sem() 计算的标准误差？{}'.format(se == se2))
    t_manual = (dmean - popmean) / se
    print('t_statistic_manually_calculated:\t{:,.4f}'.format(t_manual))
    print(u'将t值和自由度v=n-1代入 Statistical distributions and interpreting P values\n\t'
          u'http://link.zhihu.com/?target=https%3A//www.graphpad.com/quickcalcs/distMenu/\n'
          u'中可得双尾t检验的p值为0.5450。')
    #
    # 根据自由度n-1和α查找t临界值表，计算1-α=95% 的置信水平
    #   https://www.cnblogs.com/emanlee/archive/2008/10/25/1319520.html
    #  t_statistic > t临界值t_ci 就是拒绝域。
    # ref: https://zhuanlan.zhihu.com/p/29284854
    # ref: https://zhuanlan.zhihu.com/p/36727517
    #
    t_ci = 2.262
    a, b = dmean - t_ci * se, dmean + t_ci * se
    print(u'根据自由度n-1和α查找t临界值表得到t临界值：\t{}\n计算得到95%的置信区间为：\t\t\t\t\t[{}, {}]\n'.
          format(t_ci, a, b))
    #
    # 计算效应量
    d = (dmean - popmean) / ddata.std()
    print(u'效应量:\td = {:,.4f}'.format(d))
    d_res = u'大' if abs(d) >= 0.8 else (u'中' if 0.2 < abs(d) <0.8 else u'小')
    print(u"查效应量Cohen's d绝对值和效果显著含义的对应表，可知：\t差异{}".format(d_res))

    import unicodedata
    alpha, alphacode = 0.05, unicodedata.lookup('GREEK SMALL LETTER ALPHA')
    print(u'   取{} = {}'.format(alphacode, alpha))
    result = u'因为p-value<={}, 所以拒绝原假设，这些糖的包装的平均重量不等于3.5g。'.format(alphacode) \
        if p_value <= alpha \
        else u'因为p-value>{}, 所以不拒绝原假设，即这些糖的包装的平均重量等于3.5g。'.format(alphacode)
    print(u'4. 得出结论：\033[1;31m{}\033[0m'.format(result))



    ## for binomial distribution: Bernouli Experiment
    # Example1
    # 抛硬币的例子：如果抛10次,  4次正面, 6次反面, 得到的p值为0.7539。
    # in ss.binom_test(x, n, p=0.5, alternative='two-sided'):
    #   x: the number of successes, or if x has length 2, it is the number of successes and the number of failures.
    #   n: the number of trials.  This is ignored if x gives both the number of successes and failures
    pvalue = ss.binom_test(x=(4, 6))
    print("The p-value of the two-sided hypothesis test in a Bernoulli experiment:\t{:,.4f}".format(pvalue))



# classes

# main entry
if __name__ == "__main__":
    main()
    