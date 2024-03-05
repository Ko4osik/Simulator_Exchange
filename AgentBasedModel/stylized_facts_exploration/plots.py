import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
import seaborn as sns
from sktime.utils.plotting import plot_correlations
from AgentBasedModel.stylized_facts_exploration.utils import *
from AgentBasedModel.model.features import BidAskVolumeImbalance

def plot_price_movement(info):
    plt.figure(figsize=(10, 6))
    plt.plot(info.prices[list(info.exchanges.keys())[0]])
    plt.title('Price movement during the simulation')
    plt.xlabel('tick')
    plt.ylabel('price')
    plt.show()

def plot_returns_movement(info):
    plt.figure(figsize=(10, 6))
    plt.plot(get_returns(info.prices[list(info.exchanges.keys())[0]]))
    plt.title('Returns movement during the simulation')
    plt.xlabel('tick')
    plt.ylabel('returns')
    plt.show()

def plot_correlations_returns(info):
    plot_correlations(pd.Series(get_returns(info.prices[list(info.exchanges.keys())[0]])));

def plot_raw_returns_autocorrelation(info):
    acf_values, lags, lower_bound, upper_bound = extract_info_for_time_series_acf(get_returns(get_prices(info)))
    plt.figure(figsize=(10, 6))
    plt.bar(lags[1:], acf_values[1:], width=0.2, align='center', label='ACF')
    plt.fill_between(lags[1:], lower_bound, upper_bound, color='grey', alpha=0.2, label='95% Confidence Interval')
    plt.plot(lags[1:], upper_bound, color='grey')
    plt.plot(lags[1:], lower_bound, color='grey')

    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='dotted')
    plt.title('ACF of raw returns')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_absolute_returns_autocorrelation(info):
    acf_values, lags, lower_bound, upper_bound = extract_info_for_time_series_acf(np.square(get_returns(get_prices(info))))
    plt.figure(figsize=(10, 6))
    plt.bar(lags[1:], acf_values[1:], width=0.2, align='center', label='ACF')
    plt.fill_between(lags[1:], lower_bound, upper_bound, color='grey', alpha=0.2, label='95% Confidence Interval')
    plt.plot(lags[1:], upper_bound, color='grey')
    plt.plot(lags[1:], lower_bound, color='grey')

    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='dotted')
    plt.title('ACF of absolute (squared) returns')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_trade_volume_autocorrelation(info):
    acf_values, lags, lower_bound, upper_bound = extract_info_for_time_series_acf(trades_volume(info))
    plt.figure(figsize=(10, 6))
    plt.bar(lags[1:], acf_values[1:], width=0.2, align='center', label='ACF')
    plt.fill_between(lags[1:], lower_bound, upper_bound, color='grey', alpha=0.2, label='95% Confidence Interval')
    plt.plot(lags[1:], upper_bound, color='grey')
    plt.plot(lags[1:], lower_bound, color='grey')

    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='dotted')
    plt.title('ACF of trades_volume')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_orderbook_imbalance_atocorrelation(info):    
    bid_ask_volume_imbalance = BidAskVolumeImbalance(info).compile_feature()[list(info.exchanges.keys())[0]]
    acf_values, lags, lower_bound, upper_bound = extract_info_for_time_series_acf(pd.Series(bid_ask_volume_imbalance))
    plt.figure(figsize=(10, 6))
    plt.bar(lags[1:], acf_values[1:], width=0.2, align='center', label='ACF')
    plt.fill_between(lags[1:], lower_bound, upper_bound, color='grey', alpha=0.2, label='95% Confidence Interval')
    plt.plot(lags[1:], upper_bound, color='grey')
    plt.plot(lags[1:], lower_bound, color='grey')

    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='dotted')
    plt.title('ACF of order book imbalance')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_volume_volatility_correlation(info):
    x = trades_volume(info)[1:]
    y = price_volatility(info)[1:]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.plot(x,p(x),"r--", label = f'corr = {np.corrcoef(x, y)[0, 1]}')
    plt.title('Volume/volatility correlation')
    plt.xlabel('Quantity')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()

def plot_raw_returns_volatility_correlation(info):
    x = get_returns(get_prices(info))
    y = price_volatility(info)[1:]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.plot(x,p(x),"r--", label = f'corr = {np.corrcoef(x, y)[0, 1]}')
    plt.xlabel('Raw returns/volatility correlation')
    plt.xlabel('Returns')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()

def plot_leverage_effect(infos):
    infos_corrcoef_list = []
    for i in range(10):
        corrcoef_list = []
        pv = price_volatility(infos[i])
        returns = get_returns(get_prices(infos[i]))
        for i in range(1, 11):
                corrcoef_list.append(np.corrcoef(pv[:-i - 1], returns[i:])[0, 1])
        infos_corrcoef_list.append(corrcoef_list)
    plt.figure(figsize  = (12, 6))
    plt.boxplot(infos_corrcoef_list)
    plt.xticks(range(1, 11))
    plt.xlabel('Lags')
    plt.ylabel('Correlation')
    plt.title('Leverage effect')
    plt.show()

def plot_returns_density_window(info):
    window_list = [10, 50, 200, 1000]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    returns = get_returns(pd.Series(info.prices[list(info.exchanges.keys())[0]]))
    for i, window in enumerate(window_list):
        for _ in range(29):
            start = np.random.choice(len(returns) - window)
            sns.kdeplot(returns[start:start+window], color = 'grey', linestyle = 'dotted', ax = axs[i // 2, i % 2])
        kurtosis = round(np.mean(kurtosis_rolling(info, window)), 2)
        sns.kdeplot(returns[start:start+window], color = 'grey', linestyle = 'dotted', ax = axs[i // 2, i % 2], label = f'kurtosis = {kurtosis}')
        sns.kdeplot(returns, label = 'returns', color = 'green', ax = axs[i // 2, i % 2])
        sns.kdeplot(np.random.normal(loc = np.mean(returns[start:start+window]), scale = np.std(returns[start:start+window]), size=100000), label = 'gaussian', color = 'red', ax = axs[i // 2, i % 2])
        axs[i // 2, i % 2].set_title(f'Density of returns on {window} iterations')
        axs[i // 2, i % 2].set_xlabel('Value')
        axs[i // 2, i % 2].legend()
    plt.show()

def plot_returns_density(info):
    returns = get_returns(get_prices(info))
    kurtosis = round(sts.kurtosis(returns), 2)
    plt.figure(figsize = (8, 6))
    sns.kdeplot(returns, color = 'green', label = f'returns (kurtosis = {kurtosis + 3})')
    sns.kdeplot(np.random.normal(loc = np.mean(returns), scale = np.std(returns), size=100000), label = 'gaussian', color = 'red')
    plt.legend()
    plt.plot()

def plot_price_autocorrelation(info):    
    acf_values, lags, lower_bound, upper_bound = extract_info_for_time_series_acf(get_prices(info))
    plt.figure(figsize=(10, 6))
    plt.bar(lags[1:], acf_values[1:], width=0.2, align='center', label='ACF')
    plt.fill_between(lags[1:], lower_bound, upper_bound, color='grey', alpha=0.2, label='95% Confidence Interval')
    plt.plot(lags[1:], upper_bound, color='grey')
    plt.plot(lags[1:], lower_bound, color='grey')
    
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='dotted')
    plt.title('ACF of price')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_trades_volume(info):
    plt.figure(figsize = (14, 6))
    plt.plot(trades_volume(info)[1:])
    plt.title('Trades volume during the simulation')
    plt.xlabel('iter')
    plt.ylabel('Quantity')



