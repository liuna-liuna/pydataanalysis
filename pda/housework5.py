#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework5.py

    DESCRIPTION
        读入课程资源中作业数据的所有数据集

    MODIFIED  (MM/DD/YY)
        Na  11/24/2018

"""
__VERSION__ = "1.0.0.11242018"


# imports
import numpy as np
import pandas as pd
import os, os.path, sys
import pprint as pp
import json, requests
import pandas.io.sql as sql

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week5_data')

# functions
def main():
    # print out name of all dataset files
    fnames = os.listdir(DATA_PATH)
    print('{} dataset files are found:\n\t{}'.format(len(fnames), ' '.join(fnames)))

    # read in all datasets
    # multi-index
    f_mindex = os.path.join(DATA_PATH, 'csv_mindex.csv')
    df_mindex = pd.read_csv(f_mindex, index_col=['key1', 'key2'])
    print('Read in {} as\n{}'.format(f_mindex, df_mindex))
    # other csvs
    f_ex1 = os.path.join(DATA_PATH, 'ex1.csv')
    df_ex1 = pd.read_csv(f_ex1)
    print('Read in {} as\n{}'.format(f_ex1, df_ex1))

    f_ex2 = os.path.join(DATA_PATH, 'ex2.csv')
    df_ex2 = pd.read_csv(f_ex2)
    print('Read in {} as\n{}'.format(f_ex2, df_ex2))

    f_ex3 = os.path.join(DATA_PATH, 'ex3.csv')
    df_ex3 = pd.read_csv(f_ex3, sep='\s+')
    print('Read in {} as\n{}'.format(f_ex3, df_ex3))

    f_ex3_txt = os.path.join(DATA_PATH, 'ex3.txt')
    df_ex3_txt = pd.read_table(f_ex3_txt, sep='\s+')
    print('Read in {} as\n{}'.format(f_ex3_txt, df_ex3_txt))

    f_ex4 = os.path.join(DATA_PATH, 'ex4.csv')
    df_ex4 = pd.read_csv(f_ex4, skiprows=[0, 2, 3])
    print('Read in {} as\n{}'.format(f_ex4, df_ex4))

    f_ex5 = os.path.join(DATA_PATH, 'ex5.csv')
    sentinels = {'message': ['NA', 'foo'], 'something': 'two'}
    df_ex5 = pd.read_csv(f_ex5, na_values=sentinels)
    print('Read in {} as\n{}'.format(f_ex5, df_ex5))

    f_ex6 = os.path.join(DATA_PATH, 'ex6.csv')
    chunker = pd.read_csv(f_ex6, chunksize=1000)
    N = 10
    df_ex6 = pd.Series()
    for piece in chunker:
        df_ex6 = df_ex6.add(piece['key'].value_counts(), fill_value=0)
    df_ex6.sort_values(ascending=False)
    print("Read in {}, top {} after counting and sorted based on 'key' column:\n{}".format(
        f_ex6, N, df_ex6[:N]))

    # same for ex7.csv and test_file.csv
    f_ex7 = os.path.join(DATA_PATH, 'ex7.csv')
    import csv
    header = data = None
    with open(f_ex7) as f:
        reader = csv.reader(f)
        header = reader.next()
        data = list(reader)
    df_ex7 = pd.DataFrame({k:v for k, v in zip(header, zip(*data))})
    print('Read in {} as\n{}'.format(f_ex7, df_ex7))

    # testfile
    f_testfile = os.path.join(DATA_PATH, 'test_file.csv')
    with open(f_testfile) as f:
        reader = csv.reader(f)
        contents = list(reader)
    header, data = contents[0], contents[1:]
    df_testfile = pd.DataFrame({k: v for k, v in zip(header, zip(*data))})
    print('Read in {} as\n{}'.format(f_testfile, df_testfile))

    # binary data
    f_pickle = os.path.join(DATA_PATH, 'frame_pickle')
    df_pickle = pd.read_pickle(f_pickle)
    print('Read in {} as\n{}'.format(f_pickle, df_pickle))

    # out.csv
    f_out = os.path.join(DATA_PATH, 'out.csv')
    df_out = pd.read_csv(f_out)
    new_columns = ['original_index']
    new_columns.extend(df_out.columns[1:])
    df_out.columns = new_columns
    print('Read in {} as\n{}'.format(f_out, df_out))

    # tseries.csv
    f_tseries = os.path.join(DATA_PATH, 'tseries.csv')
    df_series = pd.read_csv(f_tseries)
    print('Read in {} as\n{}'.format(f_tseries, df_series))


# classes

# main entry
if __name__ == "__main__":
    main()
    