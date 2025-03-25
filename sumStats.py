from portfolio import pchart

from monthly_returns import portfolio_returns as yearly
from portfolio_monthly import portfolio_returns as monthly

import pandas as pd
import matplotlib.pyplot as plt
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
    d['CUM_RET_x'] = d.groupby(['PORTFOLIO'])['RET_x'].transform(lambda x: np.cumprod(1 + x))
    d['CUM_RET_y'] = d.groupby(['PORTFOLIO'])['RET_y'].transform(lambda x: np.cumprod(1 + x))

    portfolios = d['PORTFOLIO'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for i, portfolio in enumerate(portfolios):
        ax = axes[i]
        portfolio_data = d[d['PORTFOLIO'] == portfolio]
        
        ax.plot(portfolio_data['DATE'], portfolio_data['CUM_RET_x'], label='Annual Rebalance')
        ax.plot(portfolio_data['DATE'], portfolio_data['CUM_RET_y'], label='Monthly Rebalance')
        
        ax.set_title(f'Portfolio Comparison {portfolio}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()