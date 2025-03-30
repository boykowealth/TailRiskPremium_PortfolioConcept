import pandas as pd
from pandas.tseries.offsets import MonthEnd

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import PercentFormatter

from monthly_returns import portfolio_returns

def benchmark():
    df = pd.read_csv(r"C:\Users\Brayden Boyko\Downloads\HistoricalPrices (2).csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df['MONTHYEAR'] = df['Date'].dt.to_period('M')
    df['AVE'] = df.groupby('MONTHYEAR')[' Close'].transform('mean')
    df = df.sort_values(by='Date', ascending=True)
    df['DATE'] = df['Date']

    df = df[['DATE', 'AVE']]

    return df

def ivol_ret():
    ivol = portfolio_returns()
    bench = benchmark()

    ivol = ivol[ivol['PORTFOLIO'].isin(['LongPosV1'])]

    ivol_v = ivol.pivot(index='DATE', columns='PORTFOLIO', values='IVOL').reset_index()
    ivol_v['IVOL'] = ivol_v['LongPosV1']
    ivol_v['IVOL'] = np.log(ivol_v['IVOL'] / ivol_v['IVOL'].shift(1)) * 10
    ivol_v = ivol_v[['DATE', 'IVOL']]

    ivol_r = ivol.pivot(index='DATE', columns='PORTFOLIO', values='RET').reset_index()
    ivol_r['RET'] = ivol_r['LongPosV1']
    ivol_r = ivol_r[['DATE', 'RET']]

    df = pd.merge(ivol_v, ivol_r, on='DATE', how='left')
    df = df[(df['DATE'] >= "2007-01-01") & (df['DATE'] <= "2021-12-31")]

    df = pd.merge(df, bench, on='DATE', how='left')
    df['AVE'] = np.log(df['AVE'] / df['AVE'].shift(1))
    df['IRET'] = np.log(1 + df['RET'])

    subsets = [
        df[(df['IRET'] <= 0.1) & (df['IRET'] >= 0) & (df['IVOL'] <= 0) & (df['IVOL'] >= -0.5)],
        df[(df['IRET'] <= 0) & (df['IRET'] >= -0.1) & (df['IVOL'] <= 0.05) & (df['IVOL'] >= 0)]
    ]

    titles = [
        "Idiosyncratic Volatility Changes vs. Long Portfolio Returns (Positive Returns)",
        "Idiosyncratic Volatility Changes vs. Long Portfolio Returns (Negative Returns)"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='#1D1D1B')

    for i, (subset, title) in enumerate(zip(subsets, titles)):
        ax1 = axes[i, 0]
        ax1.set_facecolor('#1D1D1B')
        ax1.hist(subset['IVOL'], bins=20, label='Idiosyncratic Volatility', color='red', alpha=0.5, stacked=True)
        ax1.hist(subset['IRET'], bins=20, label='Returns', color='green', alpha=0.5, stacked=True)

        ax1.set_xlabel("Value", color='white')
        ax1.set_ylabel("Frequency", color='white')
        ax1.set_title(title, fontsize=10, color='white', loc='left')
        ax1.legend(facecolor='#1D1D1B', edgecolor='white', fontsize=8, labelcolor='white')
        ax1.grid(True, linestyle=':', color='white')

        ax2 = axes[i, 1]
        ax2.set_facecolor('#1D1D1B')
        ax2.scatter(subset['IRET'], subset['IVOL'], color='green', alpha=0.6)

        slope, intercept, r_value, _, _ = linregress(subset['IRET'], subset['IVOL'])
        regression_line = slope * subset['IRET'] + intercept

        sorted_IRET = np.sort(subset['IRET'])
        coeffs = Polynomial.fit(subset['IRET'], subset['IVOL'], 2).convert().coef
        polynomial_fit_sorted = coeffs[0] + coeffs[1] * sorted_IRET + coeffs[2] * sorted_IRET**2

        y_actual = subset['IVOL']
        y_predicted = coeffs[0] + coeffs[1] * subset['IRET'] + coeffs[2] * (subset['IRET'] ** 2)
        r_squared = 1 - (((y_actual - y_predicted) ** 2).sum() / ((y_actual - y_actual.mean()) ** 2).sum())

        ax2.plot(subset['IRET'], regression_line, color='red', label=f"Regression (R²={r_value**2:.2f})")
        ax2.plot(sorted_IRET, polynomial_fit_sorted, color='orange', label=f"Polynomial (R²={r_squared:.2f})")

        ax2.set_xlabel("Log Return", color='white')
        ax2.set_ylabel("Log Change In Idiosyncratic Volatility", color='white')
        ax2.set_title("Relationship Between IVOL and Returns", fontsize=10, color='white', loc='left')
        ax2.legend(facecolor='#1D1D1B', edgecolor='white', fontsize=8, labelcolor='white')
        ax2.grid(True, linestyle=':', color='white')

        ax2.yaxis.set_major_formatter(PercentFormatter(1))

    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1))
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        ax.tick_params(axis='x', colors='white') 
        ax.tick_params(axis='y', colors='white') 

    plt.tight_layout()
    plt.show()

ivol_ret()