from portfolio import pchart

from monthly_returns import portfolio_returns as yearly
from portfolio_monthly import portfolio_returns as monthly

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mtick
import numpy as np

def tickers():

    data = pchart()

    data = data[['Year', 'LongPosV2', 'ShortPosV2']]
    exploded_long = data.explode('LongPosV2').rename(columns={'LongPosV2': 'Ticker'}).assign(Portfolio='Long')
    exploded_short = data.explode('ShortPosV2').rename(columns={'ShortPosV2': 'Ticker'}).assign(Portfolio='Short')

    data = pd.concat([exploded_long, exploded_short], ignore_index=True)
    data = data[['Year', 'Portfolio', 'Ticker']]

    return data

def monthly_rebalance_compare():
    dY = yearly()
    dM = monthly()

    dY = dY[['DATE', 'PORTFOLIO', 'RET']]
    dM = dM[['DATE', 'PORTFOLIO', 'RET']]

    d = pd.merge(dY, dM, how='left', on=['DATE', 'PORTFOLIO'])
    d = d[['DATE', 'PORTFOLIO', 'RET_x', 'RET_y']]
    d = d[d['DATE'] >= "2007-01-01"]
    d['CUM_RET_x'] = d.groupby(['PORTFOLIO'])['RET_x'].transform(lambda x: np.cumprod(1 + x))
    d['CUM_RET_y'] = d.groupby(['PORTFOLIO'])['RET_y'].transform(lambda x: np.cumprod(1 + x))

    portfolios = d['PORTFOLIO'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    fig.patch.set_facecolor('#1D1D1B')

    for i, portfolio in enumerate(portfolios):
        ax = axes[i]
        portfolio_data = d[d['PORTFOLIO'] == portfolio]

        print(portfolio_data)
        
        ax.plot(portfolio_data['DATE'], portfolio_data['CUM_RET_x'], label='Annual Rebalance', color='#9fc3cf')
        ax.plot(portfolio_data['DATE'], portfolio_data['CUM_RET_y'], label='Monthly Rebalance', color='#5e91a4')

        ax.set_title(f'Portfolio Comparison {portfolio}', fontsize=12, color='white', loc='left')
        ax.set_xlabel("Date", fontsize=10, color='white')
        ax.set_ylabel("Cumulative Return", fontsize=10, color='white')

        # Formatting ticks, legend, and grid
        ax.tick_params(axis='x', labelsize=8, colors='white')
        ax.tick_params(axis='y', labelsize=8, colors='white')
        
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

        ax.legend(facecolor='#1D1D1B', edgecolor='white', fontsize=10, labelcolor='white')
        ax.grid(True, linestyle=':', color='white')

        # Axis and formatting
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        ax.set_facecolor('#1D1D1B')

    plt.tight_layout()
    plt.show()

monthly_rebalance_compare()

def htmlTable(path):

    df = pd.read_csv(path)
    df = df.to_html(index=False, float_format="{:.4f}".format)

    return df
