from monthly_returns import portfolio_returns
from portfolio import pchart
from sumStats import tickers

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
    portfolio = portfolio_returns()
    portfolio['CUM_RET'] = portfolio.groupby(['PORTFOLIO'])['RET'].transform(lambda x: np.cumprod(1 + x))
    portfolio['MONTH'] = portfolio['DATE'].dt.month
    portfolio['YEAR'] = portfolio['DATE'].dt.year
    portfolio = portfolio[portfolio['MONTH'] == 12]
    portfolio['INVESTMENT'] = portfolio['CUM_RET'] * 100_000

    portfolio_v1 = portfolio[portfolio['PORTFOLIO'].str.contains('Long')]
    portfolio_v2 = portfolio[portfolio['PORTFOLIO'].str.contains('Short')]

    pivot_v1 = portfolio_v1.pivot(index='YEAR', columns='PORTFOLIO', values='INVESTMENT')
    pivot_v2 = portfolio_v2.pivot(index='YEAR', columns='PORTFOLIO', values='INVESTMENT')

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharey=True)
    width = 0.4 ## BAR WIDTH

    for ax, pivot, title in zip(axes, [pivot_v1, pivot_v2], ["Long Portfolios", "Short Portfolios"]):
        years = pivot.index
        portfolios = pivot.columns

        for i, port in enumerate(portfolios):
            ax.bar(years + (i - len(portfolios)/2) * width, pivot[port], width=width, label=port)

        ax.set_xlabel("Year")
        ax.set_title(title)
        ax.set_xticks(years)
        ax.legend(title="Portfolio")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_ylabel("Portfolio Value ($)")
    plt.tight_layout()
    plt.show()


def ticker_count():
    data = tickers()
    data['Position'] = data.groupby(['Year', 'Portfolio']).cumcount() + 1
    data['Portfolio_Value'] = data['Portfolio'].map({'Long': 1, 'Short': -1})
    data['plot'] = data['Position'] * data['Portfolio_Value']
    
    plt.figure(figsize=(8, 8))

    for _, row in data.iterrows():
        plt.scatter(row['Year'], row['plot'], s=100, 
                    color='white' if row['Portfolio'] == 'Long' else 'white')

        plt.text(row['Year'], row['plot'], row['Ticker'], 
                 ha='center', va='center', fontsize=5, rotation=0)

    plt.axhline(0, color='black', linewidth=1) 
    plt.xticks(sorted(data['Year'].unique()))
    plt.yticks([])
    plt.xlabel("Year")
    plt.ylabel("Short (Bottom) vs Long (Top)")
    plt.title("Ticker Portfolio Positions by Year")
    plt.show()