#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        dataprocessing_practice.py

    DESCRIPTION
        to process data and pre-process data
        mainly 5 points:
        1. data cleaning
            for missing value, for exceptional value:
                delete;
                intepolation, fillna etc.:
                  mean / median / most-frequent-value,
                  fixed value, knn, most-smiliar-object-value,
                  regression, intepolation;
                no-processing

                lagrange:
                1、很多人可能觉得样本数据越多，得到的插值数据会越精确，这样想法是不正确的。
                    理论上说，样本数据过多，得到的插值函数的次数就越高，插值的结果的误差可能会更大。
                    拉格朗日插值的稳定性不太好，出现不稳定的现象称为龙格现象，解决的办法就是分段用较低次数的插值多项式。
                2、插值一般采用内插法，也就是只计算样本点内部的数据。
                作者：Black先森
                链接：https://www.jianshu.com/p/c7d6b9db8b56
                from scipy.interpolate import lagrange

        2. data integration
            pd.merge, pd.join, np.concatinate, pd.concat
                pd.merge, pd.join: data have same index / columns, to be integrated into 1 record;
                pd.concat:         data have similar structure, to be integrated into records by their own.
                pd.merge has many options:
                    on, how, left_on, right_on, left_index, right_index,
                    suffixes, sort, copy;
                    how = inner, outer, left, right
                    use which options, e.g. left_index, right_index at the same time depends on data,
                        otherwise strange dataset generated.
                df1.join(...):
                  similar to pd.merge with actually in left-outer mechanism
                      vice-verse right-outer mechanism when it's feasible
                  have only on, how, sort, lsuffix, rsuffix options, by default outer join on index;
                  concatenate only without using only 1 result for same index / column / key.
                    # concatenate: axis=...:
                np.concatenate:
                for pd.Series and pd.DataFrame: pd.concat:
                    by default axis=0, join='outer',
                    options: objs, axis, join, join_axes, keys, levels, names,
                         sort, ignore_index, verify_integrity
                    for pd.Series:
                     join_axes assigns the index, keys assigns column
                        join_axes must be [[...]], keys [...]
                    for pd.DataFrame:
                        keys assign MultiIndex, for columns or for index depends on axis=....
                        names assign names for keys
                        join_axes is less used in pd.DataFrame.

        3. data transformation
            use data in another np.ndarray, pd.Series, pd.DataFrame to replace np.nan
            duplicates processing:
                df1.drop_duplicates(subsets, keep='first', inplace=False)
            data replacement:
                df.replace(on1, nn1)
                df.replace([on1, on2], [nn1, nn2])
                df.replace((on1, on2), (nn1, nn2))
                df.replace({on1: nn1, on2: nn2})
            permutation and random sampling
                np.random.permutation(N) returns 1d np.ndarray of randomly permutation of N.
            indicator/dummy variables
                pd.get_dummies: convert categorical variable into dummy/indicator variables
                df['data1'] is a Series while df[['data1']] is a DataFrame
            add a new_column:
                df1[new_column] = ...:                        df1 is updated inplace
                df2.assign(A=func/existing_value_of_df1...):  df1 unchanged, df2 is added a A column.
            save data to excel:
                pd.read_excel(io, sheet_name=...) read in data from by default 'Sheet1'
                df1.to_excel(excel_writer, sheet_name=...): if directly use file name as excel_writer, overwrite;
                                                            if use with pd.ExcelWriter(filename) as writer:
                                                                        df1.to_excel(...); df2.to_excel(...)
                                                                , non-overwriting.
                1）读取 excel 文档的时候， 用 sheet_name 指定读取哪个 Sheet 表格
                   pd.read_excel(io, sheet_name=...) read in data from by default 'Sheet1'
                2）保存数据到 excel 的时候，
                   如果要覆盖数据，或者 excel 文件不存在的时候，
                      df1.to_excel(excel_writer, sheet_name=...): if directly use file name as excel_writer, overwrite;
                   如果要保留原来数据，例如原来数据 df1 在 ‘Sheet1’ 表格里，把新数据 df2 写入 ‘Sheet2’ 表格：
                      with pd.ExcelWriter(filename) as writer:
                           df1.to_excel(writer)
                          df2.to_excel(writer, sheet_name='Sheet2')


        4. data reshape and by-axis-rotation
            convert between long- and wide-format
              1) df2 = DataFrame(df1, columns=[...]); df2.index = periods.to_timestamp('D', 'end') works;
                  df2 = DataFrame(df1.to_records(), columns=[...], index=[...]) works;
                      while (df1, columns=[...], index=periods.to_timestamp('D', 'end')) not.

                  <=> df2 = DataFrame(df1, columns=[...], index=[...]), columns, index are used to retrieve data
                        and df1.to_recrods() is <class 'numpy.recarray'>.

              2) .pivot(<index>, <column>, <value>) = .set_index(<index>, <column>).unstack(<to_column>)
                  where .unstack(...) steps did sorting kinda of sort_index().
                          .unstack(level=..., fill_value=...), level = -1, fill_value=np.nan by default.
                  # to get data with specific condition, and when the result is 1 line and has 8 columns
                  data[(data.sex == 'Female') & (data['size'] == 5)].stack()

        5. string processing
            例如： 如何理解 get_maximum = lambda x: x.xs(x.value.idxmax(x))
            对每一个输入的x， 类型是 DataFrame，取其 value 较大的那一行

            方法签名：
            df.xs(key, axis=0, level=None, drop_level=True): cross-section (row(s) or column(s)) from the Series/DataFrame.
            # ref: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.xs.html

            相似的方法：
            DataFrame.at : Access a single value for a row/column label pair
            DataFrame.iloc : Access group of rows and columns by integer position(s)
            DataFrame.xs : Returns a cross-section (row(s) or column(s)) from the Series/DataFrame.
            Series.loc : Access group of rows and columns using label(s) or a boolean array.

            小结：
            1） df1.xs(key, axis=0, level=None, drop_level=True) is similar to df1.loc(label_or_labels_or_boolean_or_booleanarray),
            2) df1.xs is a little bit wider selection scope than def1.loc, 例如取同一level的index 中的多行的时候，用 df.loc 没法取，用 df1.xs 可以。

            # ref: https://stackoverflow.com/questions/14964493/multiindex-based-indexing-in-pandas

    MODIFIED  (MM/DD/YY)
        Na  11/27/2018

"""
__VERSION__ = "1.0.0.11272018"


# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from pandas.errors import MergeError
from scipy.interpolate import lagrange
import os, os.path
import pprint as pp

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week6_data')
# CURRENT_PATH, DATA PATH here is for running code snippet in Python Console by Shift+Alt+E
# CURRENT_PATH = os.getcwd()
# DATA_PATH = os.path.join(CURRENT_PATH, r'pda\data\week6_data')

# settings
np.set_printoptions(precision=4, threshold=500)
np.random.seed(12345)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format
# with pd.option_with('display.float_format', '{:,.4f}'.format');
# ref:  https://stackoverflow.com/questions/20937538/how-to-display-pandas-dataframe-of-floats-using-a-format-string-for-columns
plt.rc('figure', figsize=(10, 6))

# functions
def main():
    #         1. data cleaning
    #             for missing value, for exceptional value:
    #                 delete;
    #                 intepolation, fillna etc.:
    #                   mean / median / most-frequent-value,
    #                   fixed value, knn, most-smiliar-object-value,
    #                   regression, intepolation;
    #                 no-processing
    # lagrange intepolation
    fsale = os.path.join(DATA_PATH, 'catering_sale.xls')
    data = pd.read_excel(fsale)
    N = 10
    print('data samples {} read from {} in original form:\n{}'.format(N, fsale, data[:N]))
    data[u'销量'][(data[u'销量'] < 400) | (data[u'销量'] > 5000 )] = None
    print('data after removed exceptional values:\n{}'.format(data))

    # process missing value
    orig_miss_idx = None
    for j in data.columns:
        dc_nulls = data[j].isnull()
        if dc_nulls.any():
            orig_miss_idx = np.ravel(np.where(dc_nulls))    # only get the missing index for last columns
            print(u"{} missing values in data[u'{}']:\n\t{}".format(
                dc_nulls.sum(), j, orig_miss_idx))
        else:
            pass

    # [Done] 1. why lagrange returns minus value? 龙格现象，分段用较低次数的插值多项式
    # <= e.g. use k=3 instead of k=5 by default, no negative value
    def polyinterp_column(s, n, k=3):
        # y = s[n-k: n] + s[n+1: n+1+k]                 # will return a Series of NaN
        y = s[n-k:n].add(s[n+1:n+1+k], fill_value=0)    # works. fill_value = np.nan returns NaN.
        # y = s[range(n-k, n) + range(n+1, n+1+k)]        # same as y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))]
        y = y[y.notnull()]
        y_intp = lagrange(y.index, list(y))
        print('in lagrange interpolation when n = {}, y_index = {}, yn = {}:\n{}'.format(
            n, y.index, y_intp(n), np.array(y_intp)))
        return y_intp(n)
        # return lagrange(y.index, list(y))(n)

    for j in data.columns:
        for i in range(len(data)):
            if (data[j].isnull())[i]:
                # data[j][i] = polyinterp_column(data[j], i)
                data.loc[i, j] = polyinterp_column(data[j], i)
    print('After lagrange intepolation: missing values are set to:\n{}'.format(
        data[u'销量'].take(orig_miss_idx)))
    fout = os.path.join(DATA_PATH, 'sales.xls')
    data.to_excel(fout)

    # use .fillna(method=...)
    #   method='ffill': use the former value, e.g. ds[13] to fill ds[14]
    #   method='bfill': use the next value, e.g. ds[15] to fill ds[14]
    #   fillna(0):      use 0 to fill NaN
    #   fillna({'x1': 1, 'x2': 2}): use 1 for column 'x1', 2 for column 'x2'
    ds = data[u'销量']
    ds = ds.fillna(method='ffill')

    # 2. data integration
    #             pd.merge, pd.join, np.concatinate, pd.concat
    #             pd.merge has many options:
    #                   on, how, left_on, right_on, left_index, right_index,
    #                   suffixes, sort, copy;
    #                   how = inner, outer, left, right
    #
    # merge with on, how, suffixes, left_on, right_on
    df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                     'data1': range(7)})
    df2 = DataFrame({'key': ['a', 'b', 'd'],
                     'data2': range(3)})
    print('df1:\n{}\ndf2:\n{}'.format(df1, df2))
    print('By default pd will merge by same columns:\n{}'.format(pd.merge(df1, df2)))
    print('Naming the on=... to do pd.merge:\n{}'.format(pd.merge(df1, df2, on='key')))
    df11 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                      'data': range(7)})
    df22 = DataFrame({'key': ['a', 'b', 'd'],
                      'data': range(3)})
    print('Using suffixes=... instead of default _x, _y for columns with same name:\n{}'.format(
        pd.merge(df11, df22, on='key', suffixes=['_df11', '_df22'])))

    df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                     'data1': range(7)})
    df4 = DataFrame({'rkey': ['a', 'b', 'd'],
                     'data2': range(3)})
    try:
        pd.merge(df3, df4)
    except MergeError as e:
        print('df3, df4 have no common columns.\n\t{}'.format(str(e)))

    print('left_on, right_on for DataFrames with no common columns work:\n{}'.format(
        pd.merge(df3, df4, left_on='lkey', right_on='rkey')))
    # use left_index, right_index at the same time depends on data, otherwise strange dataset generated.
    print('left_on, right_on, left_index for DataFrames with no common columns work:\n{}'.format(
        pd.merge(df3, df4, left_on='lkey', right_on='rkey', left_index=True)))
    print("how='outer':\n{}".format(pd.merge(df1, df2, how='outer')))

    df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                     'data1': range(6)})
    df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                     'data2': range(5)})
    print("how='left':\n{}".format(pd.merge(df1, df2, how='left')))
    print("how='inner' which is by default:\n{}".format(pd.merge(df1, df2, how='inner')))

    left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                      'key2': ['one', 'two', 'one'],
                      'lval': [1, 2, 3]})
    right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                       'key2': ['one', 'one', 'one', 'two'],
                       'rval': [4, 5, 6, 7]})
    print("on=['key1', 'key2'], how='outer':\n{}".format(
        pd.merge(left, right, on=['key1', 'key2'], how='outer')))
    print("on='key1:\n{}".format(pd.merge(left, right, on='key1')))
    print("on='key1, suffixes=('_left', '_right'):\n{}".format(
        pd.merge(left, right, on='key1', suffixes=('_left', '_right'))))

    # merge with left_index, right_index
    left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
    right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
    print('when df1 had a column with same name as the index of df2:\n{}'.format(
        pd.merge(left1, right1, left_on='key', right_index=True)))
    print('\thow="outer":\n{}'.format(
        pd.merge(left1, right1, left_on='key', right_index=True, how='outer')))

    lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                       'key2': [2000, 2001, 2002, 2001, 2002],
                       'data': np.arange(5.)})
    righth = DataFrame(np.arange(12).reshape((6, 2)),
                       index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                              [2001, 2000, 2000, 2000, 2001, 2002]],
                       columns=['event1', 'event2'])
    print('pd.merge with left_on, right_index:\n{}'.format(
        pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)
    ))
    print('\thow="outer":\n{}'.format(
        pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer')))

    left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'],
                      columns=['Ohio', 'Nevada'])
    right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                       index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
    print('pd.merge with left_index, right_index:\n{}'.format(
        pd.merge(left2, right2, left_index=True, right_index=True)))
    print('\twith how="outer":\n{}'.format(
        pd.merge(left2, right2, left_index=True, right_index=True, how='outer')))

    # df1.join(...):
    #   similar to pd.merge with actually in left-outer mechanism
    #       vice-verse right-outer mechanism when it's feasible
    #   have only on, how, sort, lsuffix, rsuffix options
    # have only on, how, sort, lsuffix, rsuffix options, by default outer join on index;
    # concatenate only without using only 1 result for same index / column / key.
    print('df1.join(df2):\n{}'.format(left2.join(right2)))
    print('\twith how="outer":\n{}'.format(left2.join(right2, how='outer')))
    print('\tactual left-outer join:\n{}'.format(left1.join(right1, on='key')))
    print('\tright-outer join:\n{}'.format(right2.join(left2)))

    another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                        index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])
    print('df1.join([df2, df3]...):\n{}'.format(left2.join([right2, another])))
    print('\t with how="outer", right-left join:\n{}'.format(right2.join([left2, another], how='outer')))

    # concatenate: axis=...:
    #   np.concatenate,
    #   for pd.Series and pd.DataFrame: pd.concat:
    #       by default axis=0, join='outer',
    #       options: objs, axis, join, join_axes, keys, levels, names,
    #            sort, ignore_index, verify_integrity
    #       for pd.Series:
    #           join_axes assigns the index, keys assigns column
    #           join_axes must be [[...]], keys [...]
    #       for pd.DataFrame:
    #           keys assign MultiIndex, for columns or for index depends on axis=....
    #           names assign names for keys
    #           join_axes is less used in pd.DataFrame.
    arr = np.arange(12).reshape((3, 4))
    arr2 = np.random.randint(10, size=(2, 4))
    print('for np.ndarray: concatenate with axis=0;\n{}'.format(np.concatenate([arr, arr, arr2])))
    arr3 = np.random.randint(10, size=(3, 3))
    print('\twith axis=1;\n{}'.format(np.concatenate([arr, arr3], axis=1)))

    s1 = Series([0, 1], index=['a', 'b'])
    s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
    s3 = Series([5, 6], index=['f', 'g'])
    print('for pd.Series: pd.concat([s1, s2, s3]) with axis=0 by default:\n{}'.format(
        pd.concat([s1, s2, s3])))
    print('\twith axis=1:\n{}'.format(pd.concat([s1, s2, s3], axis=1)))

    s4 = pd.concat([s1 * 5, s3])
    print('Series have same index with axis=0 by default:\n{}'.format(pd.concat([s1, s4])))
    print('\twith axis=1:\n{}'.format(pd.concat([s1, s4], axis = 1)))
    print('\twith axis=1, join="outer" by default:\n{}'.format(pd.concat([s1, s4], axis=1, join='outer')))
    print('\twith axis=1, join="inner":\n{}'.format(pd.concat([s1, s4], axis=1, join='inner')))
    print('\twith axis=1, join_axes=[indexes for other n-1 axis]:\n{}'.format(
        pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])))

    result = pd.concat([s1, s2, s3], keys=['one', 'two', 'three'])
    print('for pd.Series: pd.concat([s1, s2, s3]) with axis=0, keys=...:\n{}'.format(result))
    result.unstack()

    print('for pd.Series: pd.concat([s1, s2, s3]) with axis=1, keys=...:\n{}'.format(
        pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])))

    df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                    columns=['one', 'two'])
    df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                    columns=['three', 'four'])
    print('for pd.DataFrame: pd.concat with axis=1, keys=... as MultiIndex columns:\n{}'.format(
        pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])))
    print('\tsame result as pd.concat(dict((("level1", df1), ("level2", df2))), axis=1):\n{}'.format(
        pd.concat({'level1': df1, 'level2': df2}, axis=1)))
    print('for pd.DataFrame: pd.concat keys=... as MultiIndex index\n:{}'.format(
        pd.concat([df1, df2], sort=False, keys=['level1', 'level2'])))
    print('for pd.DataFrame, pd.concat with keys as MultiIndex for columns, names for MultiIndex:\n{}'.format(
        pd.concat([df1, df2], axis=1, keys=['level1', 'level2'], names=['upper', 'lower'])))
    print('\tpd.concat with keys as MultiIndex for index, names for MultiIndex:\n{}'.format(
        pd.concat([df1, df2], sort=False, keys=['level1', 'level2'], names=['upper', 'lower'])))

    df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
    df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
    print('for pd.DataFrame: pd.concat(...) when dfs have same index:\n{}'.format(
        pd.concat([df1, df2])))
    print('\twith ignore_index=True:\n{}'.format(pd.concat([df1, df2], ignore_index=True)))
    print('\twith join="inner":\n{}'.format(pd.concat([df1, df2], ignore_index=True, join='inner')))
    print('\twith axis=1:\n{}'.format(pd.concat([df1, df2], axis=1)))

    # use data in another np.ndarray, pd.Series, pd.DataFrame to replace np.nan
    a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
               index=['f', 'e', 'd', 'c', 'b', 'a'])
    b = Series(np.arange(len(a), dtype=np.float64),
               index=['f', 'e', 'd', 'c', 'b', 'a'])
    b[-1] = np.nan
    print('use data in b to replace nan in a:\nuse np.where to generate a np.ndarray:\n{}'.
          format(np.where(pd.isnull(a), b, a)))
    print('\tsame result via a.combine_first(b), to generate a pd.Series:\n{}'.
          format(a.combine_first(b)))
    print('partial Series with combine_first:\n{}'.format(b[:-2].combine_first(a[2:])))

    df1 = DataFrame({'a': [1., np.nan, 5., np.nan],
                     'b': [np.nan, 2., np.nan, 6.],
                     'c': range(2, 18, 4)})
    df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.],
                     'b': [np.nan, 3., 4., 6., 8.]})
    print('for pd.DataFrame: use combine_first to replace nan:\n{}'.format(
        df1.combine_first(df2)))

    # stack, unstack MultiIndex
    data = DataFrame(np.arange(6).reshape((2, 3)),
                     index=pd.Index(['Ohio', 'Colorado'], name='state'),
                     columns=pd.Index(['one', 'two', 'three'], name='number'))
    result = data.stack()
    print('df1.stack():\n{}'.format(result))
    print('df1.unstack() the innest level by default:\n{}'.format(result.unstack()))
    print('df1.unstack(0):\n{}'.format(result.unstack(0)))
    print('df1.unstack(<name>):\n{}'.format(result.unstack('state')))

    s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
    s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
    # convert to MultiIndex via keys as index
    data2 = pd.concat([s1, s2], keys=['one', 'two'])
    # .unstack() set missing value as np.nan
    data2.unstack()
    # .stack() removes np.nan by default
    data2.unstack().stack()
    # use .stack(dropna=False): extends dfs to same shape with np.nan as missing value
    data2.unstack().stack(dropna=False)

    df = DataFrame({'left': result, 'right': result + 5},
                   columns=pd.Index(['left', 'right'], name='side'))
    print('unstack for df with MultiIndex as index and Index as columns:\n{}'.format(
        df.unstack('state')))
    df.unstack('state').stack('side')

    # convert between long- and wide-format
    #   1) df2 = DataFrame(df1, columns=[...]); df2.index = periods.to_timestamp('D', 'end') works;
    #       df2 = DataFrame(df1.to_records(), columns=[...], index=[...]) works;
    #           while (df1, columns=[...], index=periods.to_timestamp('D', 'end')) not.
    #       <=> df2 = DataFrame(df1, columns=[...], index=[...]), columns, index are used to retrieve data
    #   2) .pivot(<index>, <column>, <value>) = .set_index(<index>, <column>).unstack(<to_column>)
    #       where .unstack(...) steps did sorting kinda of sort_index().
    #               .unstack(level=..., fill_value=...), level = -1, fill_value=np.nan by default.
    #
    fmacro = os.path.join(DATA_PATH, r'macrodata.csv')
    data_orig = pd.read_csv(fmacro)
    periods = pd.PeriodIndex(year=data_orig.year, quarter=data_orig.quarter, name='date')
    data = DataFrame(data_orig.to_records(),
                     columns=pd.Index(['realgdp', 'infl', 'unemp'], name='item'),
                     index=periods.to_timestamp('D', 'end'))

    ldata = data.stack().reset_index().rename(columns={0: 'value'})
    print('long-format after df1.stack().reset_index().rename(columns=...):\n{}'.
        format(ldata))
    wdata = ldata.pivot('date', 'item', 'value')
    print('wide-format after ldata.pivot(...):\n{}'.format(wdata))

    N = 10
    print('ldata[:{}] before adding new column:\n{}'.format(N, ldata[:N]))
    pivoted = ldata.pivot('date', 'item', 'value')
    print('ldata.pivot(...).head():\n{}'.format(pivoted.head()))

    ldata['value2'] = np.random.randn(len(ldata))
    print('ldata[:{}] after adding new column:\n{}'.format(N, ldata[:N]))

    pivoted = ldata.pivot('date', 'item')
    print('ldata.pivot(...)[:{}] for partial columns:\n{}'.format(N, pivoted[:N]))
    print('\t1 column of ldata.pivot(...)[:{}]\n:{}'.format(N, pivoted['value'][:N]))

    unstacked = ldata.set_index(['date', 'item']).unstack('item')
    print('ldata.set_index(...).unstack(...)[:{}]:\n{}'.format(N, unstacked[:N]))

    # duplicates processing
    #   df1.drop_duplicates(subsets, keep='first', inplace=False)
    #
    data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                      'k2': [1, 1, 2, 3, 3, 4, 4]})
    print('df data is duplicated?\n{}'.format(data.duplicated()))
    print('after drop_duplicates:\n{}'.format(data.drop_duplicates()))
    data['v1'] = range(data.shape[0])
    print('added 1 new column to df:\n{}'.format(data))
    print('drop_duplicates on subsets, keep="first" by default:\n{}'.format(
        data.drop_duplicates(['k1'])))
    print('drop_duplicates on subsets, keep="last":\n{}'.format(
        data.drop_duplicates(['k1', 'k2'], keep='last')))
    print('drop_duplicates on subsets, keep=False:\n{}'.format(
        data.drop_duplicates(['k1', 'k2'], keep=False)))

    # data transformation: map, function
    data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                               'corned beef', 'Bacon', 'pastrami', 'honey ham',
                               'nova lox'],
                      'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
    meat_to_animal = {
        'bacon': 'pig',
        'pulled pork': 'pig',
        'pastrami': 'cow',
        'corned beef': 'cow',
        'honey ham': 'pig',
        'nova lox': 'salmon'
    }
    data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
    print('Data after adding 1 new column with map:\n{}'.format(data))
    print('\tanother method to transform data:\n{}'.format(
        data['food'].map(lambda x: meat_to_animal[x.lower()])))

    # data scaling
    fnorm = os.path.join(DATA_PATH, r'normalization_data.xls')
    data = pd.read_excel(fnorm, header=None)
    print('data read from {}:\n{}'.format(fnorm, data))

    print('MinMaxScaler:\n{}'.format((data - data.min()) / (data.max() - data.min()) ))
    print('StandardScaler (MeanStdScaler):\n{}'.format((data - data.mean()) / data.std() ))
    print('MaxAbsScaler (DecimalScaler):\n{}'.format(data / 10 ** np.ceil(np.log10(data.abs().max())) ))

    # data replacement:
    #   df.replace(on1, nn1)
    #   df.replace([on1, on2], [nn1, nn2])
    #   df.replace((on1, on2), (nn1, nn2))
    #   df.replace({on1: nn1, on2: nn2})
    #
    data = Series([1., -999., 2., -999., -1000., 3.])
    print('original data:\n{}'.format(data))
    print('replace(-999, np.nan):\n{}'.format(data.replace(-999, np.nan)))
    print('replace([-999, -1000], np.nan):\n{}'.format(data.replace([-999, -1000], np.nan)))
    print('replace((-999, -1000), (np.nan, 0) ):\n{}'.format(data.replace((-999, -1000), (np.nan, 0))))
    print('replace works with dict too:\n{}'.format(data.replace({-999: np.nan, -1000: 0})))

    # rename index, columns
    data = DataFrame(np.arange(12).reshape((3, 4)),
                     index=['Ohio', 'Colorado', 'New York'],
                     columns=['one', 'two', 'three', 'four'])
    data.index = data.index.map(str.upper)
    print('after index.map(...):\n{}'.format(data))
    print('after rename(index=..., columns=...):\n{}'.format(data.rename(index=str.upper, columns=str.upper)))
    print('after partial rename(index=..., columns=...):\n{}'.format(
        data.rename(index={'OHIO': 'ohio'}, columns={'three': 'THREE'})))
    _ = data.rename(index={'OHIO': 'ohio'}, inplace=True)
    print('returns ref to df always with inplace=True:\n{}'.format(data))

    # discretization and binning
    ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
    bins = [18, 25, 35, 60, 100]
    cats = pd.cut(ages, bins)
    print('discretization via pd.cut(...):\n{}'.format(cats))
    print('cats.categories:\n\t{}'.format(cats.categories))
    print('pd.value_counts(cats):\n\t{}'.format(pd.value_counts(cats)))

    print('pd.cut(..., right=False):\n{}'.format(pd.cut(ages, np.add(bins, 1), right=False)))

    group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
    print('pd.cut(..., labels=...):\n{}'.format(pd.cut(ages, bins, labels=group_names)))

    data = np.random.rand(20)
    print('pd.cut(..., 4, precision=2) for np.random.rand(...):\n{}'.format(
        pd.cut(data, 4, precision=2) ))
    print('data.max:\t{:,.2f}\ndata.min:\t{:,.2f}\n(max - min) / 4:\t{:,.2f}'.format(
        data.max(), data.min(), (data.max() - data.min()) / 4 ))

    data = np.random.randn(1000)  # Normally distributed
    cats = pd.qcut(data, 4, precision=2)  # Cut into quartiles
    print('pd.qcut(..., 4) for np.random.randn(...):\n{}'.format(cats))
    print('count in each bin:\n{}'.format(pd.value_counts(cats)))
    print('\tsame result via cats.value_counts():\n{}'.format(cats.value_counts()))
    print('np.info(data):\n{}'.format(np.info(data)))
    print('np.quantile(data, [0, 0.25, 0.75, 1]):\n{}'.format(
        np.quantile(data, [0, 0.25, 0.75, 1])))

    cats2 = pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
    print('pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.]):\n{}'.format(cats))
    print('count in each bin:\n{}'.format(pd.value_counts(cats2)))

    # exception
    np.random.seed(12345)
    data = DataFrame(np.random.randn(1000, 4))
    print('data.describe:\n{}'.format(data.describe()))

    col = data[3]
    print('np.abs(data[3]) > 3:\n{}'.format(col[np.abs(col) > 3]))
    print('in all data, abs(...) > 3:\n{}'.format( data[(np.abs(data) > 3).any(1)] ))

    # set exception value
    data[np.abs(data) > 3] = np.sign(data) * 3
    print('data.describe after set exception value:\n{}'.format(data.describe()))

    data.clip(-2, 2, inplace=True)
    print('data.describe after data.clip(..., inplace=True):\n{}'.format(data.describe()))
    data_eq2 = data[np.abs(data) == 2].dropna(thresh=1)
    print('{} lines have np.abs(number) == 2 after data.clip(-2, 2):\n{}'.format(data_eq2.shape[0], data_eq2))

    # permutation and random sampling
    #   np.random.permutation(N) returns 1d np.ndarray of randomly permutation of N.
    df = DataFrame(np.arange(5 * 4).reshape((5, 4)))
    sampler = np.random.permutation(5)
    print('sampler via np.random.permutation(5):\n{}'.format(sampler))
    print('df:\n{}'.format(df))
    print('df.take(sampler):\n{}'.format(df.take(sampler)))

    print('take partial data:\n{}'.format(df.take(np.random.permutation(len(df))[:3])))

    bag = np.array([5, 7, -1, 6, 4])
    sampler = np.random.randint(0, len(bag), size=10)
    print('sampler via np.random.randint:\n{}'.format(sampler))
    draws = bag.take(sampler)
    print('draws via bag.take(sampler):\n{}'.format(draws))

    # indicator/dummy variables
    #   pd.get_dummies: convert categorical variable into dummy/indicator variables
    #   df['data1'] is a Series while df[['data1']] is a DataFrame
    #
    df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                    'data1': range(6)})
    print('pd.get_dummies(df["key"]):\n{}'.format(pd.get_dummies(df['key'])))

    dummies = pd.get_dummies(df['key'], prefix='key')
    print('dummies:\n{}'.format(dummies))
    df_with_dummy = df[['data1']].join(dummies)
    print('df[...].join(dummies):\n{}'.format(df_with_dummy))

    f_movie = os.path.join(DATA_PATH, r'movies.dat')
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table(f_movie, sep='::', header=None, names=mnames)
    N = 10
    print('movies top {}:\n{}'.format(N, movies[:N]))

    genre_iter = (set(x.split('|')) for x in movies.genres)
    genres = sorted(set.union(*genre_iter))
    print('put genres into set and sorted:\n{}'.format(genres))

    dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)
    print('top {} dummies of movies, genres initiated:\n{}'.format(N, dummies[:N]))
    for i, gen in enumerate(movies.genres):
        dummies.ix[i, gen.split('|')] = 1
    print('top {} dummies of movies, genres after setting genres:\n{}'.format(N, dummies[:N]))
    movies_windic = movies.join(dummies.add_prefix('Genre_'))
    print('top {} movies.join with dummies.add_prefix(...):\n{}'.format(N, movies_windic[:N]))
    print('First line of movies.join with dummies.add_prefix(...):\n{}'.format(movies_windic.ix[0]))

    values = np.random.rand(N)
    print('original .rand({}) values:\n{}'.format(N, values))
    bins = np.linspace(0, 1, 6)
    dummies = pd.get_dummies(pd.cut(values, bins))
    print('dummies after pd.cut(...):\n{}'.format(dummies))

    # 线损率属性构造
    #   df1[new_column] = ...:                        df1 is updated inplace
    #   df2.assign(A=func/existing_value_of_df1...):  df1 unchanged, df2 is added a A column.
    #   pd.read_excel(io, sheet_name=...) read in data from by default 'Sheet1'
    #   df1.to_excel(excel_writer, sheet_name=...): if directly use file name as excel_writer, overwrite;
    #                                               if use with pd.ExcelWriter(filename) as writer:
    #                                                           df1.to_excel(...); df2.to_excel(...)
    #                                                   , non-overwriting.
    #
    fin = os.path.join(DATA_PATH, r'electricity_data.xls')
    fout = fin
    data = data_orig = pd.read_excel(fin)       # by default 'Sheet1'
    # data = pd.read_excel(fin)       # by default 'Sheet1'
    print('read in data in {}:\n{}'.format(fin, data))
    # calculate line_loss_rate
    data = data_orig.assign(new_col_name=lambda x: (x[u'供入电量'] - x[u'供出电量']) / x[u'供入电量'])
    data.rename(columns={u'new_col_name': u'线损率'}, inplace=True)
    # data[u'线损率'] = (data[u'供入电量'] - data[u'供出电量']) / data[u'供入电量']
    print(u'计算了线损率以后:\n{}'.format(data))
    # save data
    # following 2 lines: the later will overwrite the former
    # data_orig.to_excel(fout, index=False)
    # data.to_excel(fout, sheet_name='Sheet2', index=False)   # write to 'Sheet2'

    # non-overwriting with pd.ExcelWriter: both ways work.
    # writer = pd.ExcelWriter(fout)
    # data_orig.to_excel(writer, index=False)
    # data.to_excel(writer, sheet_name='Sheet2', index=False)
    # writer.close()
    with pd.ExcelWriter(fout) as writer:
        data_orig.to_excel(writer, index=False)
        data.to_excel(writer, sheet_name='Sheet2', index=False)

    # str
    val = 'a,b,  guido'
    val.split(',')
    pieces = [x.strip() for x in val.split(',')]
    first, second, third = pieces
    first + '::' + second + '::' + third
    '::'.join(pieces)
    'guido' in val
    val.index(',')
    val.find(':')   # -1
    val.index(':')  # ValueError: ...
    val.count('a')
    val.replace(',', '::')
    val.replace(',', '')

    # re
    import re
    text = "foo    bar\t baz  \tqux"
    re.split(r'\s+', text)
    regex1 = re.compile('\s+')
    regex1.findall(text)

    text = """Dave dave@google.com
    Steve steve@gmail.com
    Rob rob@gmail.com
    Ryan ryan@yahoo.com
    """
    pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
    regex1 = re.compile(pattern, flags=re.I)
    regex1.findall(text)
    m = regex1.search(text)
    print('The first email address was found:\n{}'.format(text[m.start(): m.end()]))
    print('use regex.match(text), found?\t{}'.format(regex1.match(text)))
    print('regex.sub(...):\n{}'.format(regex1.sub('REDACTED', text)))

    # groups()
    pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
    regex1 = re.compile(pattern, flags=re.I)
    m = regex1.match('wesm@bright.net')
    print('found result to m.groups():\n{}'.format(m.groups()))
    print('re.findall:\n{}'.format(regex1.findall(text)))
    print('After replaced:\n{}'.format(regex1.sub(r'Username: \1, Domain: \2, Suffix: \3', text)))

    regex1 = re.compile("""
        (?P<username>[A-Z0-9._%+-]+)
        @
        (?P<domain>[A-Z0-9.-]+)
        \.
        (?P<suffix>[A-Z]{2,4})""", flags=re.IGNORECASE | re.VERBOSE)

    m = regex1.match('wesm@bright.net')
    print('in m.groupdict() format:\n{}'.format(m.groupdict()))

    # vectorized str in pd
    data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
            'Rob': 'rob@gmail.com', 'Wes': np.nan}
    data = Series(data)
    print('data.isnull()?\n{}'.format(data.isnull()))
    print('data.str.contains("gmail")?\n{}'.format(data.str.contains('gmail')))
    print('data.str.findall(...):\n{}'.format(data.str.findall(pattern, flags=re.I)))
    matches = data.str.match(pattern, flags=re.I)
    print('matches?\n{}'.format(matches))

    # data is Series, matches is also a Series
    print('type(matches):\t{}'.format(type(matches)))
    print('matches.str.get(1):\n{}'.format(matches.str.get(1)))
    print('matches.str[0]:\n{}'.format(matches.str[0]))
    print('data.str[:5]:\n{}'.format(data.str[:5]))
    print('matches.str:\n{}'.format(matches.str))

    # Example: USDA food dataset
    '''
    {
      "id": 21441,
      "description": "KENTUCKY FRIED CHICKEN, Fried Chicken, EXTRA CRISPY,
    Wing, meat and skin with breading",
      "tags": ["KFC"],
      "manufacturer": "Kentucky Fried Chicken",
      "group": "Fast Foods",
      "portions": [
        {
          "amount": 1,
          "unit": "wing, with skin",
          "grams": 68.0
        },

        ...
      ],
      "nutrients": [
        {
          "value": 20.8,
          "units": "g",
          "description": "Protein",
          "group": "Composition"
        },

        ...
      ]
    }
    '''

    import json
    fjson= os.path.join(DATA_PATH, r'foods-2011-10-03.json')
    with open(fjson) as f:
        db = json.load(f)
    print('read in data from {}, len:\n\t{}'.format(fjson, len(db)))
    print('sample of data:\n\tdb[0].keys:\t{}\n\tdb[0]["nutrients"][0]:\t{}'.format(
        db[0].keys(), db[0]['nutrients'][0]))

    nutrients = DataFrame(db[0]['nutrients'])
    N = 7
    print('top {} of db[0]["nutrients"]:\n{}'.format(N, nutrients[:N]))
    info_keys = ['description', 'group', 'id', 'manufacturer']
    info = DataFrame(db, columns=info_keys)
    print('top {} of db[0][...info...]:\n{}'.format(N, info[:N]))
    print('value_counts(...) of info:\n{}'.format(pd.value_counts(info.group)[:N]))

    nutrients = []
    for rec in db:
        fnuts = DataFrame(rec['nutrients'])
        fnuts['id'] = rec['id']
        nutrients.append(fnuts)
    nutrients = pd.concat(nutrients, ignore_index=True)
    print('top {} of nutrients after pd.concat:\n{}'.format(N, nutrients[:N]))

    print('nutrients.duplicated()?\n{}'.format(nutrients.duplicated().sum()))
    col_mapping = {'description': 'food',
                   'group': 'fgroup'}
    info = info.rename(columns=col_mapping, copy=False)
    print('top {} of info after rename:\n{}'.format(N, info[:N]))

    col_mapping = {'description': 'nutrient',
                   'group': 'nutgroup'}
    nutrients = nutrients.rename(columns=col_mapping, copy=False)
    ndata = pd.merge(nutrients, info, on='id', how='outer')
    print('top {} of pd.merge(nutrients, info)...\n{}'.format(N, ndata[:N]))
    print('30000 line:\n{}'.format(ndata.ix[30000]))

    result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
    print('top {} of top 50% value after .groupby(...):\n{}'.format(N, result[:N]))

    result['Zinc, Zn'].sort_values().plot(kind='barh')
    print('result of Adjusted Protein:\n{}'.format(result['Adjusted Protein']))

    by_nutrient = ndata.groupby(['nutgroup', 'nutrient'])
    # df1.xs(key, axis=0, level=None, drop_level=True): cross-section (row(s) or column(s)) from the Series/DataFrame.
    #   https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.xs.html
    get_maximum = lambda x: x.xs(x.value.idxmax())
    get_minimum = lambda x: x.xs(x.value.idxmin())
    max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]
    max_foods.rename(columns={'description': 'food'}, inplace=True)
    max_foods.food = max_foods.food.str[:50]
    max_foods.ix['Amino Acids']['food']

# classes

# main entry
if __name__ == "__main__":
    main()
    