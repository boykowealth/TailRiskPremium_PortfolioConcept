from monthly_returns import portfolio_returns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def monthly_ret():
    portfolio = portfolio_returns()

    plt.figure(figsize=(12, 6))
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

    plt.figure(figsize=(12, 6))
    for port in portfolio['PORTFOLIO'].unique():
        data = portfolio[portfolio['PORTFOLIO'] == port]
        plt.plot(data['DATE'], data['IVOL'], label=port)

    plt.xlabel("Date")
    plt.ylabel("IVOL")
    plt.title("Monthly Portfolio Idiosyncratic Volatility Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def monthly_total_return():

    portfolio = portfolio_returns()
    portfolio['CUM_RET'] = (1 + portfolio['RET']).cumprod()

    plt.figure(figsize=(12, 6))
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

    portfolio['TYPE'] = np.where(portfolio['PORTFOLIO'].isin(['LongPosV1', 'LongPosV2']), 'LONG', 'SHORT')
    alpha = portfolio.groupby(['DATE', 'TYPE'])['RET'].apply(
        RET=lambda x: x.diff()
    ).reset_index()
    portfolio = portfolio.merge(alpha, how='left', on='DATE')

    print(portfolio)


monthly_alpha()