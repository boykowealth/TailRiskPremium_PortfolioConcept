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
    portfolio['CUM_RET'] = portfolio.groupby(['PORTFOLIO'])['RET'].transform(lambda x: np.cumprod(1 + x))

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
    portfolio = portfolio[['DATE', 'PORTFOLIO', 'RET']]
    portfolio = portfolio.pivot(index='DATE', columns='PORTFOLIO', values='RET')

    portfolio['ALPHA_L'] = portfolio['LongPosV2'] - portfolio['LongPosV1']
    portfolio['ALPHA_S'] = portfolio['ShortPosV2'] - portfolio['ShortPosV1']

    portfolio['ALPHA_L'] = (1 + portfolio['ALPHA_L']).cumprod()
    portfolio['ALPHA_S'] = (1 + portfolio['ALPHA_S']).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio.index, portfolio['ALPHA_L'], label='Alpha Long')
    plt.plot(portfolio.index, portfolio['ALPHA_S'], label='Alpha Short')

    plt.xlabel("Date")
    plt.ylabel("Alpha")
    plt.title("Monthly Portfolio Alpha Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


monthly_total_return()