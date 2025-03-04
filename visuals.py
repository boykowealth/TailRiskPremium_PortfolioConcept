from monthly_returns import portfolio_returns
from portfolio import pchart

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def monthly_ret():
    portfolio = portfolio_returns()

    plt.figure(figsize=(8, 5))
    for port in portfolio['PORTFOLIO'].unique():
        data = portfolio[portfolio['PORTFOLIO'] == port]
        plt.plot(data['DATE'], data['RET'], label=port)

    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.title("Monthly Portfolio Returns Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def monthly_ivol():
    portfolio = portfolio_returns()
    portfolio = portfolio[portfolio['PORTFOLIO'].isin(['LongPosV1', 'ShortPosV1'])]
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    
    for i, port in enumerate(portfolio['PORTFOLIO'].unique()):
        data = portfolio[portfolio['PORTFOLIO'] == port]
        axs[i].plot(data['DATE'], data['IVOL'], label=port)
        axs[i].set_xlabel("Date")
        axs[i].set_ylabel("IVOL")
        axs[i].legend()
        axs[i].grid(True)
    
    plt.show()

def monthly_total_return():
    portfolio = portfolio_returns()
    portfolio['CUM_RET'] = portfolio.groupby(['PORTFOLIO'])['RET'].transform(lambda x: np.cumprod(1 + x))

    plt.figure(figsize=(8, 5))
    for port in portfolio['PORTFOLIO'].unique():
        data = portfolio[portfolio['PORTFOLIO'] == port]
        plt.plot(data['DATE'], data['CUM_RET'], label=port)

    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.title("Monthly Portfolio Returns Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def monthly_alpha():
    portfolio = portfolio_returns()
    portfolio['CUM_RET'] = portfolio.groupby(['PORTFOLIO'])['RET'].transform(lambda x: np.cumprod(1 + x))
    portfolio = portfolio[['DATE', 'PORTFOLIO', 'CUM_RET']]
    portfolio = portfolio.pivot(index='DATE', columns='PORTFOLIO', values='CUM_RET')

    portfolio['ALPHA_L'] = portfolio['LongPosV2'] - portfolio['LongPosV1']
    portfolio['ALPHA_S'] = portfolio['ShortPosV2'] - portfolio['ShortPosV1']
    
    plt.figure(figsize=(8, 5))
    plt.plot(portfolio.index, portfolio['ALPHA_L'], label='Alpha Long')
    plt.plot(portfolio.index, portfolio['ALPHA_S'], label='Alpha Short')

    plt.xlabel("Date")
    plt.ylabel("Alpha")
    plt.title("Monthly Portfolio Alpha Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def monthly_hist():
    portfolio = portfolio_returns()
    unique_ports = portfolio['PORTFOLIO'].unique()
    num_ports = len(unique_ports)
    num_rows = num_cols = 2

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 8))
    axs = axs.flatten()

    for ax, port in zip(axs, unique_ports):
        data = portfolio[portfolio['PORTFOLIO'] == port]
        ax.hist(data['RET'], bins=20, alpha=0.5, label=port, density=True)

        mean, std_dev = norm.fit(data['RET'])
        
        x = np.linspace(min(data['RET']), max(data['RET']), 100)
        p = norm.pdf(x, mean, std_dev)
        ax.plot(x, p, 'k', linewidth=2, label='Normal dist.')

        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.set_title("")
        ax.grid(True)
        ax.legend()

    plt.show()


def annual_ret():
    data = pchart()
    data = data[['Year', 'LongV1Ret', 'LongV2Ret', 'ShortV1Ret', 'ShortV2Ret']]
    data = data.melt(id_vars=['Year'], var_name='PORTFOLIO', value_name='RET')

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    portfolios = ['LongV1Ret', 'LongV2Ret', 'ShortV1Ret', 'ShortV2Ret']
    titles = ['Long V1 Returns', 'Long V2 Returns', 'Short V1 Returns', 'Short V2 Returns']

    for ax, portfolio, title in zip(axs.flatten(), portfolios, titles):
        subset = data[data['PORTFOLIO'] == portfolio]
        ax.bar(subset['Year'], subset['RET'], color='teal')
        ax.set_title(title)
        ax.set_ylabel('Return')
        ax.set_xlabel('Year')
        ax.tick_params(axis='x', rotation=45)

    plt.show()
