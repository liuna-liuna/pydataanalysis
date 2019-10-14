#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        groupby_practice.py

    DESCRIPTION
        1. groupby:
            分组键5种：
                列表或数组，其长度与待分组的轴一样， np.array with same length
                表示 DataFrame 某个列名的值, columns；如果索引或者列是多层索引，可以根据索引级别分组, level
                字典或 Series， 给出待分组轴上的值与分组名之间的对应关系， dict, Series
                函数，用于处理轴索引或索引中的各个标签，indexing, function
                df.dtypes

            1) groupby:
                if no axis=... is given, axis=0 by default.
                df = DataFrame(..., index=pd.MultiIndex.fromarrays([[...]], names=['cty',...]))
                    .groupby('key1'), .groupby(level='cty', axis=1)
                    .groupby(len), .groupby(lambda g: g.find('e'))
                    .groupby(df.dtypes, axis=1)
                    .groupby(mapping), .groupby(map_s)
                    .groupby(..., group_keys=False)

            2) first groupby then get data via columns or index is more frequently used :
                    type: pandas.core.groupby.groupby.DataFrameGroupBy
                while if in reverse order:
                    type: pandas.core.groupby.groupby.SeriesGroupBy
                df.groupby(...)[[...]]: pandas.core.groupby.groupby.DataFrameGroupBy
                df.groupby(...)[...]:   pandas.core.groupby.groupby.SeriesGroupBy
                wide data format is more frequently used:
                    groupby data via categories
                    reshape to categories as index, cand_nm as columns：
                        dfec_mrbo.groupby(['cand_nm', labels].size().unstack(0)
                        dfec_mrbo.groupby(['cand_nm', 'contbr_st'].contb_receipt_amt.sum().unstack(0).fillna(0)

            3) DataFrameGroupBy has methods: mean etc., first, head, pct_change, plot, hist etc.
                                has properties: groups, indices, ngroups, ndim etc.

            4) to get data in ...GroupBy object:
                 via dict(list(...))
                 iterate over groups
                 column can be access by .<column>.value_counts()
                 grouped.count() gives M rows x N columns info for index, all columns
                 grouped.size() gives size of each group, similar to grouped.count(), w/o listing all columns
                 grouped.describe() gives aggregation info for numeric columns

        2. aggregation： 任何能够从数组产生标量值的数据转换过程
            # 1 function, columns-oriented multiple functions
            .quantile(0.9), .describe()

            agg:    .agg('mean'), .agg(np.max), .agg(['min', np.var, peak_to_peak]),
                    .agg([('name1', 'func1')...]), .agg({'namen': ['func1', ...'funcn'])

            # transform keeps original index => sometimes same value for rows within same group
            #   e.g. when .transform(np.mean)
            <=> could use groupby again after transform
            transform 产生一个标量值 或者 一个大小相同的数组
            transform:  .transform(demean), .transform(np.mean)

            map: Series specific, to each element
                    dfec.cand_nm.map(parties), .map(lambda x: emp_mapping.get(x,x))

            apply 一般的拆分-应用-合并 Split-Apply-Combine (SAC)
            apply: apply a function to each group after groupby(...)
                    .apply(lambda g: g.describe())
                    .apply(get_stats, arg1=...).unstack()

            applymap: DataFrame specific, to each element
                    example TODO

        3. pivot_table 透视表根据一个或多个键对数据进行聚合，并根据行和列上的分组键将数据分配到各个矩形区域内
            df.pivot_table(index=..., columns=..., ...,
                           values=..., aggfunc='sum', fill_value=0, margins=True, dropna=True, margins_name='All')
            df.pivot_table(index=...), df.pivot_table(columns=...)
                index, columns: at least 1 group keys is mandatory

            DataFrame.pivot_table :
                generalization of pivot that can handle duplicate values for one index/column pair.
                Create a spreadsheet-style pivot table as a DataFrame.

                The levels in the pivot table will be stored in MultiIndex objects (hierarchical
                indexes) on the index and columns of the result DataFrame

                Parameters
                ----------
                values : column to aggregate, optional
                index : column, Grouper, array, or list of the previous
                    If an array is passed, it must be the same length as the data. The
                    list can contain any of the other types (except list).
                    Keys to group by on the pivot table index.  If an array is passed,
                    it is being used as the same manner as column values.
                columns : column, Grouper, array, or list of the previous
                    If an array is passed, it must be the same length as the data. The
                    list can contain any of the other types (except list).
                    Keys to group by on the pivot table column.  If an array is passed,
                    it is being used as the same manner as column values.
                aggfunc : function, list of functions, dict, default numpy.mean
                    If list of functions passed, the resulting pivot table will have
                    hierarchical columns whose top level are the function names
                    (inferred from the function objects themselves)
                    If dict is passed, the key is column to aggregate and value
                    is function or list of functions
                fill_value : scalar, default None
                    Value to replace missing values with
                margins : boolean, default False
                    Add all row / columns (e.g. for subtotal / grand totals)
                dropna : boolean, default True
                    Do not include columns whose entries are all NaN
                margins_name : string, default 'All'
                    Name of the row / column that will contain the totals
                    when margins is True.
            # ----------------------------------------------------------------------
            # Data reshaping
            ldata.columns: Index([u'date', u'item', u'value'], dtype='object')
            wdata = ldata.pivot('date', 'item', 'value')
                    => <class 'pandas.core.frame.DataFrame'>，
                       wdata.columns: Index([u'infl', u'realgdp', u'unemp'], dtype='object', name=u'item')
            wd2 = ldata.pivot('date', 'item')
                    => <class 'pandas.core.frame.DataFrame'>，
                       wd2.columns: MultiIndex(levels=[[u'value'], [u'infl', u'realgdp', u'unemp']],
                                               labels=[[0, 0, 0], [0, 1, 2]],
                                               names=[None, u'item'])
            ldata.pivot('date', 'item', 'value') is correct format, ldata.pivot(['date'], 'item', 'value') not.

            <=> for df.pivot(...), index, columns: both are mandatory
                for columns in type Index or MultiIndex, only level[0] has attribute shortcut.

            DataFrame.pivot : pivot without aggregation that can handle non-numeric data
                              A ValueError is raised if there are any duplicates.
                              For finer-tuned control, see hierarchical indexing documentation along
                                with the related stack/unstack methods.
            def pivot(self, index=None, columns=None, values=None):
                Return reshaped DataFrame organized by given index / column values.

                Reshape data (produce a "pivot" table) based on column values. Uses
                unique values from specified `index` / `columns` to form axes of the
                resulting DataFrame. This function does not support data
                aggregation, multiple values will result in a MultiIndex in the
                columns.

                Parameters
                ----------
                index : string or object, optional
                    Column to use to make new frame's index. If None, uses
                    existing index.
                columns : string or object
                    Column to use to make new frame's columns.
                values : string, object or a list of the previous, optional
                    Column(s) to use for populating new frame's values. If not
                    specified, all remaining columns will be used and the result will
                    have hierarchically indexed columns.

            # ----------------------------------------------------------------------
            DataFrame.unstack : pivot based on the index values instead of a column.

            .pivot(<index>, <column>, <value>) = .set_index(<index>, <column>).unstack(<to_column>)
                  where .unstack(...) steps did sorting kinda of sort_index().
                          .unstack(level=..., fill_value=...), level = -1, fill_value=np.nan by default.

        4. cross_table 交叉表一种用于计算分组频率的特殊透视表
            pd.crosstab(index, columns, ..., values=..., margins=True)
                    index, columns: both are mandatory
            pd.crosstab instead of df.crosstab.

            pd.crosstab:
                Compute a simple cross-tabulation of two (or more) factors.
                By default computes a frequency table of the factors unless an array of values and an
                aggregation function are passed
                Parameters
                ----------
                index : array-like, Series, or list of arrays/Series
                    Values to group by in the rows
                columns : array-like, Series, or list of arrays/Series
                    Values to group by in the columns
                values : array-like, optional
                    Array of values to aggregate according to the factors.
                    Requires `aggfunc` be specified.
                aggfunc : function, optional
                    If specified, requires `values` be specified as well
                rownames : sequence, default None
                    If passed, must match number of row arrays passed
                colnames : sequence, default None
                    If passed, must match number of column arrays passed
                margins : boolean, default False
                    Add row/column margins (subtotals)
                margins_name : string, default 'All'
                    Name of the row / column that will contain the totals
                    when margins is True.
                    .. versionadded:: 0.21.0
                dropna : boolean, default True
                    Do not include columns whose entries are all NaN
                normalize : boolean, {'all', 'index', 'columns'}, or {0,1}, default False
                    Normalize by dividing all values by the sum of values.
                    - If passed 'all' or `True`, will normalize over all values.
                    - If passed 'index' will normalize over each row.
                    - If passed 'columns' will normalize over each column.
                    - If margins is `True`, will also normalize margin values.
                    .. versionadded:: 0.18.1
                Notes
                -----
                Any Series passed will have their name attributes used unless row or column
                names for the cross-tabulation are specified.
                Any input passed containing Categorical data will have **all** of its
                categories included in the cross-tabulation, even if the actual data does
                not contain any instances of a particular category.
                In the event that there aren't overlapping indexes an empty DataFrame will
                be returned.
            Returns
                -------
                crosstab : DataFrame

        5. pd.qcut(df, nbins, labels=True):
                .dtypes = CategoricalDtype(categories=[(...],...], ordered=True),
                has .cat attribute.
           pd.qcut(df, nbins, labels=False):
                .dtypes = dtype('int64'), use 0...nbins-1 interval[float64] as replacement of labels.
           both are <class 'pandas.core.series.Series'>,
           both .value_counts() return data of nbins rows, both .index returns RangeIndex(...).
           which original data belongs to what category can be checked via grouping.values, collections.Counter(...)

        6. .fillna(VALUE), .fillna(s.mean), .fillna(lambda g: ..., inplace=True)
            s.fillna(s.mean(), inplace=False)<=> s has no change, return a new Series;
            s.fillna(s.mean(), inplace=True)<=> s is updated, return None.

        7. weighted averages: np.average(data, weights=...)
           corrcoef for each group:
             s.corr(other, method=..., ...): Compute correlation with `other` Series, excluding missing values
                                             method by default 'pearson'.
                                                 pearson: standard correlation coefficient,
                                                 kendall: Kendall Tau correlation coefficient,
                                                 spearman: rank Spearman correlation,
             df.corr(method=..., ...): Compute pairwise correlation of columns, excluding NA/null values

             df.corrwith(other, axis=..., drop=...): Compute pairwise correlation between rows or columns
                                                     of two DataFrame objects
                 other: DataFrame, Series
                 axis: by default 0, drop: by default False, returns union of all.
                 Returns: correls: Series.

        8.  to delete one row of df:    df.drop(labels, ...)
            to delete one column of df: df.pop(<column_name>)
            to delete the firs 3 rows of df:    df.iloc[3:], df.drop(label, inplace=True),
                                                df.drop(df.index[:3], inplace=True),
                                                df.drop(df.head(3).index, inplace=True)

            # ref: https://stackoverflow.com/questions/16396903/delete-the-first-three-rows-of-a-dataframe-in-pandas

    MODIFIED  (MM/DD/YY)
        Na  12/13/2018

"""
__VERSION__ = "1.0.0.12132018"


# imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import os
import pprint as pp

# configuration
np.set_printoptions(precision=4, threshold=500)
np.random.seed(12345)
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:,.4f}'.format
plt.rc('figure', figsize=(10, 6))
font_options = {
    'family': 'monospace',
    'size': 7,
    'weight': 'bold'
}
plt.rc('font', **font_options)

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week8_data')
# CURRENT_PATH, DATA PATH here is for running code snippet in Python Console by Shift+Alt+E
# CURRENT_PATH = os.getcwd()
# DATA_PATH = os.path.join(CURRENT_PATH, r'pda\data\week8_data')

# functions
def main():
    # groupby
    df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                    'key2': ['one', 'two', 'one', 'two', 'one'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})
    grouped = df['data1'].groupby(df['key1'])
    print("type(df[...].groupedby(...)):\t{}".format(type(grouped)))
    print("groupby(df['key1']).value_counts:\n{}".format(grouped.value_counts()))

    means = df['data1'].groupby([df['key1'], df['key2']]).mean()
    print("mean() of df[...].groupby([...]):\n{}".format(means))
    print("\t.mean().unstack():\n{}".format(means.unstack()))

    states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
    years = np.array([2005, 2005, 2006, 2005, 2006])
    means_sy = df['data1'].groupby([states, years]).mean()
    print("mean() of ...groupby... via np.array:\n{}".format(means_sy))
    print("df.groupby([states, years]).mean():\n{}".format(df.groupby([states, years]).mean()))

    # same result, following format is more frequently used
    print("df.groupby('key1').mean():\n{}".format(df.groupby('key1').mean()))
    grouped = df.groupby(['key1', 'key2'])
    print("df.groupby(['key1', 'key2']).mean():\n{}".format(grouped.mean()))
    print("df.groupby(['key1', 'key2']).size():\n{}".format(grouped.size()))

    # iterate over groups
    #   type: <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x0A702550>
    print("type(df.groupby('key1')): \t{}".format(df.groupby('key1')))
    for name, group in df.groupby('key1'):
        print('\nname:\t{}\ngroup:\n{}'.format(name, group))

    #   type: <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x0A01C1B0>
    for (k1, k2), group in df.groupby(['key1', 'key2']):
        print('\nkeys:\t{}\ngroup:\n{}'.format((k1, k2), group))

    # convert groupby.DataFrameGroupBy object to dict
    pieces = dict(list(df.groupby('key1')))
    print("dict(list(df.groupby('key1')))['b']:\n{}".format(pieces['b']))

    # groupby via dtypes
    grouped = df.groupby(df.dtypes, axis=1)
    print("dict(list(df.groupby(df.dtypes, axis=1))):\n{}".format(dict(list(grouped))))

    # select a column after groupby: DataFrameGroupBy object
    #   or groupby after select a column: SeriesGroupBy object
    print("df.groupby('key1')['data1']:\n{}".format(df.groupby('key1')['data1']))
    print("df['data1'].groupby(df['key1']):\n{}\n".format(df['data1'].groupby(df['key1'])))
    print("df.groupby('key1')[['data2']]:\n{}".format(df.groupby('key1')[['data2']]))
    print("df[['data2']].groupby(df['key1']):\n{}\n".format(df[['data2']].groupby(df['key1'])))

    # df.groupby(...)[[...]]: pandas.core.groupby.groupby.DataFrameGroupBy
    # df.groupby(...)[...]:   pandas.core.groupby.groupby.SeriesGroupBy
    df_grouped = df.groupby(['key1', 'key2'])[['data2']]
    print("df.groupby(['key1', 'key2'])[['data2']].mean():\n{}\n".format(df_grouped.mean()))
    print("df.groupby(['key1', 'key2'])[['data2']] is a {}".format(df_grouped))

    s_grouped = df.groupby(['key1', 'key2'])['data2']
    print("df.groupby(['key1', 'key2'])['data2'] is a {}".format(s_grouped))

    # groupby via dict, Series
    people = DataFrame(np.random.randn(5, 5),
                       columns=['a', 'b', 'c', 'd', 'e'],
                       index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
    people.ix[2:3, ['b', 'c']] = np.nan
    print('people:\n{}'.format(people))
    mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
               'd': 'blue', 'e': 'red', 'f': 'orange'}
    by_column = people.groupby(mapping, axis=1)
    print("groupby via dict: people.groupby(mapping, axis=1):\n\t{}".format(by_column))
    print("\t.sum():\n{}".format(by_column.sum()))
    map_series = Series(mapping)
    print("map_series:\n{}".format(map_series))
    by_column_s = people.groupby(map_series, axis=1)
    print("groupby via Series: people.groupby(map_series, axis=1):\n\t{}".format(
        by_column_s))
    print("\t.count():\n{}".format(by_column_s.count()))

    # groupby via function
    by_func = people.groupby(len)
    print("groupby via function: people.groupby(len)\n\t{}\n\t.sum:\n{}".
          format(by_func, by_func.sum()))
    print("people.groupby(lambda g: g.find('e')):")
    pp.pprint(dict(list(people.groupby(lambda g: g.find('e')))))

    key_list = ['one', 'one', 'one', 'two', 'two']
    by_func_m = people.groupby([len, key_list])
    print('groupby via function and list: people.groupby([len, key_list]:\n{}'.
          format(by_func_m))
    print('\t.min():\n{}'.format(by_func_m.min()))

    # groupby via indexing
    columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                         [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
    hier_df = DataFrame(np.random.randn(4, 5), columns=columns)
    print('hier_df with MultiIndex as columns:\n{}'.format(hier_df))
    by_index = hier_df.groupby(level='cty', axis=1)
    print("groupby via indexing: hier_df.groupby(level='cty', axis=1):\n{}".format(by_index))
    print("\t.count():\n{}".format(by_index.count()))
    print("data in DataFrameGroupBy via dict(list(...)):")
    pp.pprint(dict(list(by_index)))

    # data aggregation
    df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                    'key2': ['one', 'two', 'one', 'two', 'one'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})
    grouped = df.groupby('key1')
    print("df.groupby('key1'):")
    pp.pprint(dict(list(grouped)))
    print("grouped['data1'].quantile(0.9):\n{}".format(grouped['data1'].quantile(0.9)))

    def peak_to_peak(arr):
        return arr.max() - arr.min()
    print("grouped.agg(peak_to_peak):\n{}".format(grouped.agg(peak_to_peak)))
    print("grouped.describe():\n{}".format(grouped.describe()))
    print("\t.describe() for each g:\n{}".format(grouped.apply(lambda g: g.describe())))

    # columns-oriented multiple functions
    ftips = os.path.join(DATA_PATH, r'tips.csv')
    tips = pd.read_csv(ftips)
    tips['tip_pct'] = tips['tip'] / tips['total_bill']
    N = 6
    print('top {} of tips:\n{}'.format(N, tips[:N]))

    grouped = tips.groupby(['sex', 'smoker'])
    grouped_pct = grouped['tip_pct']
    print("grouped_pct.agg('mean'):\n{}".format(grouped_pct.agg('mean')))
    print("grouped_pct.agg(['mean', 'std', peak_to_peak]):\n{}".
          format(grouped_pct.agg(['mean', 'std', peak_to_peak])))
    print("naming groupby(...)[...].agg(...): grouped_pct.agg([('foo', 'mean'), ('bar', np.std)]:\n{}".
          format(grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])))

    functions = ['count', 'mean', 'max']
    grouped_pct_tb = grouped['tip_pct', 'total_bill']
    result = grouped_pct_tb.agg(functions)
    print("grouped['tip_pct', 'total_bill'].agg(functions):\n{}".format(result))
    print("\tresult['tip_pct']:\n{}".format(result['tip_pct']))

    func_tuples = [('foo', 'mean'), ('bar', np.var)]
    print(".agg via list of tuples: grouped_pct_tb.agg(func_tuples):\n{}".
          format(grouped_pct_tb.agg(func_tuples)))
    func_dict = {'tip': np.max, 'size': 'sum'}
    print(".add via dict: grouped.agg(func_dict):\n{}".
          format(grouped.agg(func_dict)))
    func_dict = {'tip_pct': ['min', 'max', 'mean', 'std'],
                 'size': 'sum'}
    print("\tmultiple functions: grouped.agg(func_dict):\n{}".
          format(grouped.agg(func_dict)))
    print("grouped.describe():\n{}".format(grouped.describe()))

    # transform based on groupby
    k1_means = df.groupby('key1').mean().add_prefix('mean_')
    print("df.groupby('key1').mean().add_prefix('mean_'):\n{}".format(k1_means))
    result = pd.merge(df, k1_means, left_on='key1', right_index=True)
    print("pd.merge(df, k1_means, left_on='key1', right_index=True):\n{}".format(result))

    key = ['one', 'two', 'one', 'two', 'one']
    print("people.groupby(key).mean():\n{}".format(people.groupby(key).mean()))
    print("\t1 line same result via people.ix[np.where(pd.Series(key) == 'one')].mean():\n{}".
          format(people.ix[np.where(pd.Series(key) == 'one')].mean()))
    print("people.groupby(key).transform(np.mean);\n{}".
          format(people.groupby(key).transform(np.mean)))

    def demean(arr):
        return arr - arr.mean()
    demeaned = people.groupby(key).transform(demean)
    print("people.groupby(key).transform(demean):\n{}".format(demeaned))
    print("demeaned.groupby(key).mean();\n{}".format(demeaned.groupby(key).mean()))

    # apply method
    def tiptop(df, n=5, column='tip_pct'):
        return df.sort_index(by=column)[-n:]
    print("tiptop(tips, n=6):\n{}".format(tiptop(tips, n=6)))

    print("tips.groupby('smoker').apply(tiptop):\n{}".format(
        tips.groupby('smoker').apply(tiptop)))
    print("tips.groupby(['smoker', 'day']).apply(tiptop, n=1, column='total_bill'):\n{}".format(
        tips.groupby(['smoker', 'day']).apply(tiptop, n=1, column='total_bill')))
    result = tips.groupby('smoker')['tip_pct'].describe()
    print("tips.groupby('smoker')['tip_pct'].describe():\n{}".format(result))
    print("result.unstack('smoker'):\n{}".format(result.unstack('smoker')))

    # disable group_keys
    print("tips.groupby('smoker', group_keys=False).apply(tiptop);\n{}".
          format(tips.groupby('smoker', group_keys=False).apply(tiptop)))

    # quantile and bucket analysis
    frame = DataFrame({'data1': np.random.randn(1000),
                       'data2': np.random.randn(1000)})
    N = 5
    print("top {} of frame:\n{}".format(N, frame[:N]))

    factor = pd.cut(frame.data1, 4)
    print("top {} of pd.cut(frame.data1, 4):\n{}".format(N, factor[:N]))

    def get_stats(group):
        return {'min': group.min(), 'max': group.max(),
                'count': group.count(), 'mean': group.mean()}
    grouped = frame.data2.groupby(factor)
    print("frame.data2.groupby(factor):\n{}".format(grouped))
    print("\tdata after grouped: \n{}".format(dict(list(grouped))))
    print("grouped.apply(get_stats).unstack():\n{}".format(
        grouped.apply(get_stats).unstack()))

    # pd.qcut(df, nbins, labels=True):  .dtypes = CategoricalDtype(categories=[(...],...], ordered=True),
    #                                   has .cat attribute.
    # pd.qcut(df, nbins, labels=False): .dtypes = dtype('int64'), use 0...nbins-1 as replacement of interval[float64].
    #   both are <class 'pandas.core.series.Series'>,
    #   both .value_counts() return data of nbins rows, both .index returns RangeIndex(...).
    #   which original data belongs to what category can be checked via grouping.values, collections.Counter(...)
    #
    grouping = pd.qcut(frame.data1, 10, labels=False)
    print("\npd.qcut(frame.data1, 10, labels=False).describe():\n{}".
          format(grouping.describe()))
    grouped = frame.data2.groupby(grouping)
    print("\tdata of grouped.first():\n{}".format(grouped.first()))
    print("grouped.apply(get_stats).unstack():\n{}".format(
        grouped.apply(get_stats).unstack()))

    # fill missing values with group-specific value
    # .fillna(VALUE), .fillna(s.mean), .fillna(lambda g: ..., inplace=True)
    #   s.fillna(s.mean(), inplace=False)<=> s has no change, return a new Series;
    #   s.fillna(s.mean(), inplace=True)<=> s is updated, return None.
    #
    s = Series(np.random.randn(6))
    s[::2] = np.nan
    print("s:\n{}".format(s))
    print("s.fillna(s.mean(), inplace=False)<=> s has no change, return a new Series:\n{}".
          format(s.fillna(s.mean())))
    print("s.fillna(s.mean(), inplace=True)<=> s is updated, return None:\n{}".
          format(s.fillna(s.mean(), inplace=True)))

    states = ['Ohio', 'New York', 'Vermont', 'Florida',
              'Oregon', 'Nevada', 'California', 'Idaho']
    group_key = ['East'] * 4 + ['West'] * 4
    data = Series(np.random.randn(8), index=states)
    data[['Vermont', 'Nevada', 'Idaho']] = np.nan
    print('data:\n{}'.format(data))
    grouped = data.groupby(group_key)
    print("data.groupby(group_key).mean():\n{}".format(grouped.mean()))
    fill_mean = lambda g: g.fillna(g.mean())
    print("data.groupby(group_key).apply(fill_mean):\n{}".format(grouped.apply(fill_mean)))
    fill_values = {'East': 0.5, 'West': -1}
    fill_func = lambda g: g.fillna(fill_values[g.name])
    print("grouped.apply(fill_func);\n{}".format(grouped.apply(fill_func)))

    suits = ['H', 'S', 'C', 'D']
    card_val = (range(1, 11) + [10] * 3) * 4
    base_names = ['A'] + range(2, 11) + ['J', 'K', 'Q']
    cards = []
    for suit in suits:
        cards.extend(str(num) + suit for num in base_names)
    deck = Series(card_val, index=cards)
    print("cards data:\n{}".format(deck))

    def draw(deck, n=5):
        return deck.take(np.random.permutation(n))
        # return deck.take(np.random.permutation(len(deck))[:n])
    print("draw(deck):\n{}".format(draw(deck)))

    # disable group_keys
    get_suit = lambda card: card[-1]
    grouped = deck.groupby(get_suit, group_keys=False)
    print("grouped:\n{}".format(grouped))
    pp.pprint(dict(list(grouped)))

    result = grouped.apply(draw, n=2)
    print("deck.groupby(get_suit, group_keys=False).apply(draw, n=2):\n{}".
          format(result))

    # weight averages: np.average(data, weights=...)
    # corrcoef for each group:
    #   s.corr(other, method=..., ...): Compute correlation with `other` Series, excluding missing values
    #                                   method by default 'pearson'.
    #                                       pearson: standard correlation coefficient,
    #                                       kendall: Kendall Tau correlation coefficient,
    #                                       spearman: rank Spearman correlation,
    #   df.corr(method=..., ...): Compute pairwise correlation of columns, excluding NA/null values
    #
    #   df.corrwith(other, axis=..., drop=...): Compute pairwise correlation between rows or columns
    #                                           of two DataFrame objects
    #       other: DataFrame, Series
    #       axis: by default 0, drop: by default False, returns union of all.
    #       Returns: correls: Series.
    #
    df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                    'data': np.random.randn(8),
                    'weights': np.random.rand(8)})
    print("df:\n{}".format(df))
    grouped = df.groupby('category')
    get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
    print("grouped.apply(get_wavg):\n{}".format(grouped.apply(get_wavg)))

    fstock = os.path.join(DATA_PATH, r'stock_px.csv')
    close_px = pd.read_csv(fstock, parse_dates=True, index_col=0)
    print("close_px.info():\n{}".format(close_px.info()))
    print("close_px[-4:]:\n{}".format(close_px[-4:]))

    rets = close_px.pct_change().dropna()
    print("close_px.pct_change().dropna():\n{}".format(rets))

    spx_corr = lambda x: x.corrwith(x['SPX'])
    by_year = rets.groupby(lambda x: x.year)
    print("rets.groupby(lambda x: x.year):\n{}".format(by_year))
    pp.pprint(dict(list(by_year)))

    print("by_year.apply(spx_corr):\n{}".format(by_year.apply(spx_corr)))
    print("by_year.apply(lambda g: g['AAPL'].corr(g['MSFT'])):\n{}".
          format(by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))))
    print("\tcorr(methods='kendall'):\n{}".
          format(by_year.apply(lambda g: g['AAPL'].corr(g['MSFT'], method='kendall'))))
    print("\tcorr(methods='spearman'):\n{}".
          format(by_year.apply(lambda g: g['AAPL'].corr(g['MSFT'], method='spearman'))))

    # pivot_table
    print("tips[:{}]:\n{}".format(N, tips[:N]))
    print("tips.pivot_table(index=['sex', 'smoker']);\n{}".
          format(tips.pivot_table(index=['sex', 'smoker'])))
    print("tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'], columns='smoker'):\n{}".
          format(tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'], columns='smoker')))
    print("tips.pivot_table('tip_pct', index=['sex', 'smoker'], columns='day', aggfunc=len, margins=True):\n{}".
          format(tips.pivot_table('tip_pct', index=['sex', 'smoker'], columns='day', aggfunc=len, margins=True)))
    print("tips.pivot_table('size', index=['time', 'sex', 'smoker'], columns='day', "
          "aggfunc='sum', fill_value=0):\n{}".
          format(tips.pivot_table('size', index=['time', 'sex', 'smoker'], columns='day',
                                  aggfunc='sum', fill_value=0)))

    # cross_table
    from StringIO import StringIO
    data = """Sample    Gender    Handedness
    1    Female    Right-handed
    2    Male    Left-handed
    3    Female    Right-handed
    4    Male    Right-handed
    5    Male    Left-handed
    6    Male    Right-handed
    7    Female    Right-handed
    8    Female    Left-handed
    9    Male    Right-handed
    10    Female    Right-handed"""
    data = pd.read_table(StringIO(data), sep='\s+')
    print('data:\n{}'.format(data))

    print("pd.crosstab(data.Gender, data.Handedness, margins=True):\n{}".
          format(pd.crosstab(data.Gender, data.Handedness, margins=True)))
    print("pd.crosstab([tips.time, tips.day], tips.smoker, margins=True):\n{}".
          format(pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)))

    # analyze fec_2012_data
    ffec = os.path.join(DATA_PATH, r'P00000001-ALL.csv')
    dfec = pd.read_csv(ffec)
    print("fec.info():\n{}".format(dfec.info()))
    print("dfec.describe():\n{}".format(dfec.describe()))
    print("1 line dfec.ix[123456]:\n{}".format(dfec.ix[123456]))

    unique_cands = dfec.cand_nm.unique()
    print("unique_cands:\n{}".format(unique_cands))
    print("totally {} cands. unique_cands[2]:\t{}".format(len(unique_cands), unique_cands[2]))

    parties = {'Bachmann, Michelle': 'Republican',
               'Cain, Herman': 'Republican',
               'Gingrich, Newt': 'Republican',
               'Huntsman, Jon': 'Republican',
               'Johnson, Gary Earl': 'Republican',
               'McCotter, Thaddeus G': 'Republican',
               'Obama, Barack': 'Democrat',
               'Paul, Ron': 'Republican',
               'Pawlenty, Timothy': 'Republican',
               'Perry, Rick': 'Republican',
               "Roemer, Charles E. 'Buddy' III": 'Republican',
               'Romney, Mitt': 'Republican',
               'Santorum, Rick': 'Republican'}

    print("fec.cand_nm[123456:123461]:\n{}".format(dfec.cand_nm[123456:123461]))
    print("dfec.cand_nm[123456:123461]).map(parties):\n{}".format(
        dfec.cand_nm[123456:123461].map(parties)))

    # add new column
    # dfec.pop('parties')
    dfec['party'] = dfec.cand_nm.map(parties)
    print("dfec['party'].value_counts():\n{}".format(dfec['party'].value_counts()))

    print("dfec.contb_receipt_amt.describe():\n{}".format(dfec.contb_receipt_amt.describe()))
    print("dfec.contb_receipt_amt.isnull().any()?\t\033[1;31m{}\033[0m".format(
        dfec.contb_receipt_amt.isnull().any()))
    print("len(np.where(dfec.contb_receipt_amt < 0)[0]):\t\033[1;31m{}\033[0m".format(
        len(np.where(dfec.contb_receipt_amt < 0)[0])))
    dfec = dfec[dfec.contb_receipt_amt > 0]
    print("dfec.describe() after dfec[dfec.contb_receipt_amt > 0]:\n{}".format(dfec.describe()))

    # for 2 cands
    dfec_mrbo = dfec[dfec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]
    print("dfec_mrbo[:{}]:\n{}".format(N, dfec_mrbo[:N]))
    print("dfec_mrbo.describe():\n{}".format(dfec_mrbo.describe()))

    # do statistics based on occupation and employer
    print("dfec.contbr_occupation.value_counts()[:{}]:\n{}".format(
        N, dfec.contbr_occupation.value_counts()[:N]))

    occ_mapping = {
        'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
        'INFORMATION REQUESTED': 'NOT PROVIDED',
        'INFORMATION REQUESTED (BEST EFFORTS)': 'NOT PROVIDED',
        'C.E.O.': 'CEO'
    }
    f = lambda x: occ_mapping.get(x, x)
    dfec.contbr_occupation = dfec.contbr_occupation.map(f)

    print("dfec.contbr_employer.value_counts()[:{}]:\n{}".format(
        N, dfec.contbr_employer.value_counts()[:N]))
    emp_mapping = {
        'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
        'INFORMATION REQUESTED': 'NOT PROVIDED',
        'SELF': 'SELF-EMPLOYED',
        'SELF EMPLOYED': 'SELF-EMPLOYED',
    }
    f = lambda x: emp_mapping.get(x, x)
    dfec.contbr_employer = dfec.contbr_employer.map(f)

    # aggregation via pivot_table
    by_occupation = dfec.pivot_table('contb_receipt_amt',
                                     index='contbr_occupation',
                                     columns='party', aggfunc='sum')
    print("by_occupation[:{}]:\n{}".format(N, by_occupation[:N]))
    print("type(by_occupation):\t\033[1;31m{}\033[0m".format(type(by_occupation)))

    over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
    print("by_occupation[by_occupation.sum(1) > 2000000]:\n{}".format(over_2mm))
    print("over_2mm.describe():\n{}".format(over_2mm.describe()))
    over_2mm.plot(kind='barh')

    def get_top_amounts(group, key, n=5):
        totals = group.groupby(key)['contb_receipt_amt'].sum()
        # sort the totals by key in descending order
        return totals.sort_values(ascending=False)[:n]

    # check data in grouped:
    #   column can be access by .<column>.value_counts()
    #   grouped.count() gives M rows x N columns info for index, all columns
    #   grouped.size() gives size of each group, similar to grouped.count(), w/o listing all columns
    #   grouped.describe() gives aggregation info for numeric columns
    #
    grouped = dfec_mrbo.groupby('cand_nm')
    print("list(grouped)[0][1][:{}]:\n{}".format(N, list(grouped)[0][1][:N]))
    print("grouped.cand_id.value_counts():\n{}".format(grouped.cand_id.value_counts()))
    print("grouped.count():\n{}".format(grouped.count()))
    print("grouped.describe():\n{}".format(grouped.describe()))

    grouped.apply(get_top_amounts, 'contbr_occupation', n=7)
    grouped.apply(get_top_amounts, 'contbr_employer', n=10)

    # groupby contribution_amount via categories
    bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
    labels = pd.cut(dfec_mrbo.contb_receipt_amt, bins)
    print("pd.cut(dfec_mrbo.contb_receipt_amt, bins):\n{}".format(labels))

    grouped = dfec_mrbo.groupby(['cand_nm', labels])
    print("grouped.count():\n{}".format(grouped.count()))
    print("grouped.describe():\n{}".format(grouped.describe()))
    print("grouped.size():\n{}".format(grouped.size()))
    # reshape to categories as index, cand_nm as columns
    print("grouped.size().unstack(0:\n{}".format(grouped.size().unstack(0)))

    bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
    print("grouped.contb_receipt_amt.sum().unstack(0):\n{}".format(bucket_sums))

    normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
    print("bucket_sums.div(bucket_sums.sum(axis=1), axis=0);\n{}".format(normed_sums))
    # same plot
    normed_sums[:-2].plot(kind='barh', stacked=True)
    normed_sums.dropna().plot(kind='barh', stacked=True)

    # count contribution_amount by state
    grouped = dfec_mrbo.groupby(['cand_nm', 'contbr_st'])
    print("grouped.count():\n{}".format(grouped.count()))
    print("grouped.describe():\n{}".format(grouped.describe()))
    print("grouped.size():\n{}".format(grouped.size()))
    totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
    print("first {} of grouped.contb_receipt_amt.sum().unstack(0).fillna(0):\n{}".
          format(N, totals[:N]))
    totals = totals[totals.sum(1) > 100000]
    print("first {} of totals[totals.sum(1) > 100000]:\n{}".
          format(N, totals[:N]))
    percent = totals.div(totals.sum(1), axis=0)
    print("first {} of totals.div(totals.sum(1), axis=0):\n{}".
          format(N, percent[:N]))
    print("percent.head():\n{}".format(percent.head()))
    # percent.plot(kind='barh', stacked=True)

# classes

# main entry
if __name__ == "__main__":
    main()
