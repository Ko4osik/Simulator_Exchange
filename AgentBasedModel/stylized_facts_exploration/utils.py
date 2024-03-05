import pandas as pd
import numpy as np
import scipy.stats as sts
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import curve_fit

from statsmodels.graphics.tsaplots import acf


def get_prices(info):
    prices = info.prices[list(info.exchanges.keys())[0]]
    return prices

def get_returns(prices):
    prices = pd.Series(prices)
    return list((prices - prices.shift(1)).dropna())

def get_log_returns(prices):
    prices = pd.Series(prices)
    return list(np.log(prices / prices.shift(1)).dropna())

def kurtosis_rolling(info, window):
    returns = get_returns(get_prices(info))
    return pd.Series(returns).rolling(window = window).apply(lambda x: sts.kurtosis(x))

def trades_volume(info):
    trades_history = info.trades_history[list(info.exchanges.keys())[0]]
    trades_volume_list = []
    for tick in range(len(trades_history)):
        if len(trades_history[tick]) == 0:
            trades_volume_list.append(0)
        else:
            trades_volume_list.append(pd.DataFrame(trades_history[tick])['qty'].sum())
    return trades_volume_list

def price_volatility(info):
    trades_history = info.trades_history[list(info.exchanges.keys())[0]]
    price_volatility_list = []
    for tick in range(len(trades_history)):
        if len(trades_history[tick]) == 0:
            price_volatility_list.append(0)
        else:
            price_volatility_list.append(pd.DataFrame(trades_history[tick])['price'].std())
    return price_volatility_list

def number_of_price_changes(info):
    prices = get_prices(info)
    return ((pd.Series(prices) - pd.Series(prices).shift(1)) != 0).sum() - 1

def number_of_returns_changes(info):
    returns = get_returns(get_prices(info))
    return ((pd.Series(returns) - pd.Series(returns).shift(1)) != 0).sum() - 1

def extract_info_for_time_series_acf(time_series):
    acf_values, confint = acf(time_series, alpha=0.05)
    lower_bound = confint[1:, 0] - acf_values[1:]
    upper_bound = confint[1:, 1] - acf_values[1:]
    lags = np.arange(0, len(acf_values))
    return acf_values, lags, lower_bound, upper_bound

def ljungbox_test(times_series):
    res = sm.tsa.ARIMA(times_series, order = (1, 0, 1)).fit()
    res1 =sm.stats.acorr_ljungbox (res.resid , lags=[5], return_df= False)
    obs_value, p_value = res1['lb_stat'].item(), res1['lb_pvalue'].item()
    return obs_value, p_value

def ADF_test(times_series):
    obs_value, p_value = adfuller(times_series)[0], adfuller(times_series)[1]
    return obs_value, p_value

def kurtosis_of_returns(info):
    returns = get_returns(pd.Series(info.prices[list(info.exchanges.keys())[0]]))
    print(f"Kurtosis:", sts.kurtosis(returns))

def opt_func(t, A, b):
    return A / np.power(t, b)


def abs_returns_corr_coeffs(info):
    acf_values, lags, lower_bound, upper_bound = extract_info_for_time_series_acf(np.square(get_returns(get_prices(info))))
    n = len(acf_values)
    t = np.arange(1, n + 1)
    popt, _ = curve_fit(opt_func, t, acf_values)
    A, b = popt
    return A, b
