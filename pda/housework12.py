#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework12.py

    DESCRIPTION
        对stock_px 中的三个股票数据拟合ARIMA模型

    MODIFIED  (MM/DD/YY)
        Na  01/14/2019

"""
__VERSION__ = "1.0.0.01142019"


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
import warnings

# configuration
plt.rc('figure', figsize=(10, 6))
np.random.seed(12345)
np.set_printoptions(precision=4, threshold=500)
pd.options.display.float_format = '{:,.4f}'.format
pd.options.display.max_rows = 100
warnings.simplefilter(action='ignore', category=FutureWarning)

# consts
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, r'data\week12_data')

# functions
def check_if_stationary(ds=None, index=None, K=100, data_name_and_keyword='Original'):
    if ds is None:
        # do nothing
        return
    else:
        # check from plots
        # draw data plot, acf, pacf
        plt.figure()
        plt.plot(index, ds)
        plt.title('{} Data'.format(data_name_and_keyword))
        # draw acf, pacf only for K numbers since the data could have a too large size
        plot_acf(ds[-K:], title='{} Autocorrelation'.format(data_name_and_keyword)).show()
        plot_pacf(ds[-K:], title='{} Partial Autocorrelation'.format(data_name_and_keyword)).show()

        # check from ADF
        adf_res = ADF(ds)
        adf_output = pd.Series(data=adf_res[:4],
                               index=['t-statistics', 'p-value', '#lags used', '#observations'])
        for k, v in adf_res[4].iteritems():
            adf_output['Belief interval {}'.format(k)] = '{}'.format(v)
        # output
        print('ADF result for {} data:\n{}'.format(data_name_and_keyword, adf_output))
        # check if stationary from p-value
        if adf_output['p-value'] <= 0.05:
            print('{} data \033[1;31mis\033[0m stationary.\n'.format(data_name_and_keyword))
            return True
        else:
            print('{} data \033[1;31mis not\033[0m stationary.\n'.format(data_name_and_keyword))
            return False


def main():
    # 对stock_px 中的三个股票数据拟合ARIMA模型
    # read in data
    fstock = os.path.join(DATA_PATH, r'stock_px.csv')
    adata = pd.read_csv(fstock, index_col=0, parse_dates=[0])

    # enable plot interactive mode
    plt.ion()
    for name in adata.columns:
        # check if original data is stationary
        data_name_and_keyword, cdata = '{} Original'.format(name), adata[name]
        orig_if_stationary = check_if_stationary(cdata, index=adata.index,
                                                 data_name_and_keyword=data_name_and_keyword)
        if orig_if_stationary:
            # already stationary
            pass
        else:
            # check if data of 1-rank diff is stationary
            data_name_and_keyword, cdata = '{} 1-rank diff'.format(name), np.diff(adata[name])
            ddata_if_stationary = check_if_stationary(cdata, index=adata.index[1:],
                                                      data_name_and_keyword=data_name_and_keyword)
            if ddata_if_stationary:
                pass
            else:
                # do no checking of further-ranks
                print('\nAssume 1-rank diff is stationary, not checking diff with further-ranks.\n'
                      '=> continue with ARIMA model fitting.\n')
                pass

        # check if randomness
        diff_lb_res = acorr_ljungbox(adata[name], lags=1)
        # check if randomness from p-value
        if diff_lb_res[1][0] <= 0.05:
            print('lb_p-value={}, {} data \033[1;31mis not\033[0m pure randomness.\n=> Continue to do modeling ...\n'.
                  format(diff_lb_res[1][0], data_name_and_keyword))
        else:
            print('lb_p-value={}, {} data \033[1;31mis\033[0m pure randomness.\n=> No modeling needs to be done.'.
                  format(diff_lb_res[1][0], data_name_and_keyword))
            # no need to do modeling when pure randomness
            continue

        # ARIMA model
        # know from above, d=1; calculate p, q
        # calculate p, q via BIC only for K (here hardcoded K=3) numbers since the data could have a too large size
        # pmax = qmax = int(len(cdata) / 10)
        pmax = qmax = 3
        bic_matrix = []
        for p in xrange(pmax):
            tmp = []
            for q in xrange(qmax):
                try:
                    tmp.append(ARIMA(adata[name], (p, 1, q)).fit(disp=0).bic)
                except Exception:
                    tmp.append(None)
            bic_matrix.append(tmp)
        # find out the idxmin of bic_matrix as p, q
        bic_matrix = pd.DataFrame(bic_matrix)
        # bic_matrix = pd.DataFrame(bic_matrix).astype(np.float64)
        p, q = bic_matrix.stack().idxmin()

        # create ARIMA model
        arima_model = ARIMA(adata[name], (p, 1, q)).fit(disp=0)
        print("\nARIMA model summary for {}:\n{}\n".format(data_name_and_keyword, arima_model.summary()))

        # forecast for forecasenum days
        forecastnum = 5
        fc_res = arima_model.forecast(forecastnum)
        print("\n{} days' forecast for {}:\n{}\n".format(forecastnum, data_name_and_keyword, fc_res))

        # draw original data together with forecast data
        plt.figure()
        # draw only for K (here hardcoded K=100) numbers since the data could have a too large size
        K = 100
        plt.plot(adata.index[-K:], adata[name][-K:])
        fc_series = pd.Series(data=fc_res[0], index=adata.index[-forecastnum:].shift(forecastnum, freq='D'))
        plt.plot(fc_series)
        plt.title("{} Original Data and {} days' forecast".format(name, forecastnum))

        plt.pause(5)
        # close the figures
        plt.close('all')

# main entry
if __name__ == "__main__":
    main()
    