from monthly_returns import portfolio_returns
from portfolio import pchart
from sumStats import tickers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

def monthly_ret():
    portfolio = portfolio_returns()
    unique_ports = portfolio['PORTFOLIO'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.patch.set_facecolor('#1D1D1B')
    axes = axes.flatten() 
    for i, port in enumerate(unique_ports[:4]):  
        data = portfolio[portfolio['PORTFOLIO'] == port]
        ax = axes[i]
        ax.plot(data['DATE'], data['RET'], label=port, color='#9fc3cf')
        ax.set_title(f"Portfolio: {port}", color='white', fontsize=10, loc='left')
        ax.set_xlabel("Date", fontsize=8, color='white')
        ax.set_ylabel("Return", fontsize=8, color='white')
        ax.tick_params(axis='x', labelsize=8, colors='white')
        ax.tick_params(axis='y', labelsize=8, colors='white')
        ax.grid(True, linestyle=':', color='white')

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_facecolor('#1D1D1B') 
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

    plt.tight_layout()  # Adjust spacing between plots
    plt.show()

def monthly_ivol():
    portfolio = portfolio_returns()
    portfolio = portfolio[portfolio['PORTFOLIO'].isin(['LongPosV1', 'ShortPosV1'])]
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    fig.patch.set_facecolor('#1D1D1B')
    
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

    plt.figure(figsize=(8, 5), facecolor='#1D1D1B')
    ax = plt.gca()
    ax.set_facecolor('#1D1D1B')

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white') 
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    for port in portfolio['PORTFOLIO'].unique():
        data = portfolio[portfolio['PORTFOLIO'] == port]
        plt.plot(data['DATE'], data['CUM_RET'], label=port)

    plt.xlabel("Date", fontsize=8, color='white')
    plt.ylabel("Return", fontsize=8, color='white')
    plt.title("Monthly Portfolio Returns Over Time", fontsize=10, color='white', loc='left')
    plt.legend(facecolor='#1D1D1B',
               edgecolor='white',
               fontsize=8,
               loc='upper left',
               title='',
               title_fontsize=10,
               labelcolor='white'
            )
    plt.grid(True, linestyle=':', color='white')

    plt.show()


def monthly_alpha():
    portfolio = portfolio_returns()
    portfolio['CUM_RET'] = portfolio.groupby(['PORTFOLIO'])['RET'].transform(lambda x: np.cumprod(1 + x))
    portfolio = portfolio[['DATE', 'PORTFOLIO', 'CUM_RET']]
    portfolio = portfolio.pivot(index='DATE', columns='PORTFOLIO', values='CUM_RET')

    portfolio['ALPHA_L'] = portfolio['LongPosV2'] - portfolio['LongPosV1']
    portfolio['ALPHA_S'] = portfolio['ShortPosV2'] - portfolio['ShortPosV1']

    portfolio['ALPHA'] = portfolio['ALPHA_L'] + portfolio['ALPHA_S']
    
    plt.figure(figsize=(8, 5))
    plt.plot(portfolio.index, portfolio['ALPHA'], label='Portfolio Alpha')

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
    portfolio['INVESTMENT'] = portfolio['CUM_RET']

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

def portfolio_totals():
    portfolio = portfolio_returns()
    portfolio['CUM_RET'] = portfolio.groupby(['PORTFOLIO'])['RET'].transform(lambda x: np.cumprod(1 + x))
    v1 = portfolio[portfolio['PORTFOLIO'].str.contains('V1')]
    v2 = portfolio[portfolio['PORTFOLIO'].str.contains('V2')]
    
    ## EQUAL WEIGHT
    v1 = v1.groupby('DATE')['CUM_RET'].agg(V1='mean').reset_index()
    v2 = v2.groupby('DATE')['CUM_RET'].agg(V2='mean').reset_index()

    port = v1.merge(v2, how='left', on='DATE')

    plt.figure(figsize=(8, 5))
    plt.plot(port['DATE'], port['V1'], label='Base Portfolio')
    plt.plot(port['DATE'], port['V2'], label='Strategy Portfolio')

    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.title("Total Equal-Weighted Portfolio Returns Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    
def portfolio_skews():
    portfolio = pchart()
    port = portfolio[['Year', 'LongV1Skew', 'ShortV1Skew', 'LongV2Skew', 'ShortV2Skew']]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.patch.set_facecolor('#1D1D1B')

    skews = {
        "LongV2Skew": (0, 0),
        "ShortV2Skew": (0, 1)
    }

    for key, (row, col) in skews.items():
        ax = axes[row, col]
        ax.plot(port["Year"], port[key], marker="o", linestyle="-", label=key, color='#9fc3cf')

        ax.set_title(key, fontsize=10, color='white', loc='left')
        ax.set_xlabel("Year", fontsize=8, color='white')
        ax.set_ylabel("Skew", fontsize=8, color='white')
        ax.tick_params(axis='x', labelsize=8, colors='white')
        ax.tick_params(axis='y', labelsize=8, colors='white')
        ax.grid(True, linestyle=':', color='white')

        ax.set_facecolor('#1D1D1B')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.show()

def monthly_ivol():
    portfolio = portfolio_returns()
    portfolio = portfolio[portfolio['PORTFOLIO'].isin(['LongPosV1', 'ShortPosV1'])]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.patch.set_facecolor('#1D1D1B')

    for i, port in enumerate(portfolio['PORTFOLIO'].unique()):
        data = portfolio[portfolio['PORTFOLIO'] == port]
        ax = axes[i]
        
        ax.plot(data['DATE'], data['IVOL'], label=port, color='#9fc3cf')

        ax.set_title(f"Portfolio: {port}", fontsize=10, color='white', loc='left')
        ax.set_xlabel("Date", fontsize=8, color='white')
        ax.set_ylabel("IVOL", fontsize=8, color='white')
        ax.tick_params(axis='x', labelsize=8, colors='white')
        ax.tick_params(axis='y', labelsize=8, colors='white')
        ax.grid(True, linestyle=':', color='white')

        ax.set_facecolor('#1D1D1B')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 

    plt.tight_layout()
    plt.show()

def ticker_three_bar_chart():
    data = tickers()

    data = data[data['Year'] >= 2007]

    long_tickers = data[data['Portfolio'] == 'Long']['Ticker'].value_counts().nlargest(5)
    short_tickers = data[data['Portfolio'] == 'Short']['Ticker'].value_counts().nlargest(5)

    frequent_tickers = pd.concat([long_tickers, short_tickers], axis=1, keys=['Long', 'Short']).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#1D1D1B')
    ax.set_facecolor('#1D1D1B')

    x = range(len(frequent_tickers))
    width = 0.5

    # Plot bars
    bars_long = ax.bar([i - width / 2 for i in x], frequent_tickers['Long'], width, label='Long', color='#4A90E2')
    bars_short = ax.bar([i + width / 2 for i in x], frequent_tickers['Short'], width, label='Short', color='#9fc3cf')

    for i, bar in enumerate(bars_long):
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height() / 2, 
            frequent_tickers.index[i], 
            ha='center', va='center', color='#1D1D1B', fontsize=8
        )

    for i, bar in enumerate(bars_short):
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height() / 2, 
            frequent_tickers.index[i], 
            ha='center', va='center', color='#1D1D1B', fontsize=8
        )

    # Chart formatting
    ax.set_xticks([])
    ax.set_xticklabels([], fontsize=0, color=(1, 1, 1, 0))
    ax.set_ylabel("Frequency", fontsize=10, color='white')
    ax.set_title("Top 5 Most Frequent Tickers: Long vs. Short", fontsize=12, color='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend(facecolor='#1D1D1B', edgecolor='white', title='', title_fontsize=10, labelcolor='white')

    plt.show()

def monthly_ret_port():
    portfolio = portfolio_returns()
    
    v1 = portfolio[portfolio['PORTFOLIO'].str.contains('V1')]
    v2 = portfolio[portfolio['PORTFOLIO'].str.contains('V2')]

    v1 = v1.groupby('DATE')['RET'].agg(V1='mean').reset_index()
    v2 = v2.groupby('DATE')['RET'].agg(V2='mean').reset_index()

    port = v1.merge(v2, how='left', on='DATE')
    port = port.melt(id_vars=['DATE'], value_vars=['V1', 'V2'], var_name='Portfolio', value_name='RET')

    unique_ports = port['Portfolio'].unique()

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.patch.set_facecolor('#1D1D1B')

    axes = axes.flatten()
    for i, port_name in enumerate(unique_ports[:2]):  # Loop through unique portfolio names
        data = port[port['Portfolio'] == port_name]  # Filter correctly

        ax = axes[i]
        ax.plot(data['DATE'], data['RET'], label=port_name, color='#9fc3cf')  # Plot correct data
        ax.set_title(f"Portfolio: {port_name}", color='white', fontsize=10, loc='left')
        ax.set_xlabel("Date", fontsize=8, color='white')
        ax.set_ylabel("Return", fontsize=8, color='white')
        ax.tick_params(axis='x', labelsize=8, colors='white')
        ax.tick_params(axis='y', labelsize=8, colors='white')
        ax.grid(True, linestyle=':', color='white')

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_facecolor('#1D1D1B')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

    plt.tight_layout()
    plt.show()


def portfolio_totals_ret():
    portfolio = portfolio_returns()
    portfolio['CUM_RET'] = portfolio.groupby(['PORTFOLIO'])['RET'].transform(lambda x: np.cumprod(1 + x))
    v1 = portfolio[portfolio['PORTFOLIO'].str.contains('V1')]
    v2 = portfolio[portfolio['PORTFOLIO'].str.contains('V2')]
    
    v1 = v1.groupby('DATE')['CUM_RET'].agg(V1='mean').reset_index()
    v2 = v2.groupby('DATE')['CUM_RET'].agg(V2='mean').reset_index()

    port = v1.merge(v2, how='left', on='DATE')

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#1D1D1B')
    ax.set_facecolor('#1D1D1B')

    ax.plot(port['DATE'], port['V1'], label='Base Portfolio', color='red')
    ax.plot(port['DATE'], port['V2'], label='New Portfolio', color='green')

    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Returns", color='white')
    ax.set_title("Total Equal-Weighted Portfolio Returns Over Time", fontsize=12, color='white')
    ax.legend(facecolor='#1D1D1B', edgecolor='white', fontsize=8, labelcolor='white')
    ax.grid(True, linestyle=':', color='white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.show()


def benchmark():
    df = pd.read_csv(r"C:\Users\Brayden Boyko\Downloads\HistoricalPrices (3).csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df['MONTHYEAR'] = df['Date'].dt.to_period('M')
    df['AVE'] = df.groupby('MONTHYEAR')[' Close'].transform('mean')
    df = df.sort_values(by='Date', ascending=True)
    df['DATE'] = df['Date']

    df = df[['DATE', 'AVE']]

    return df

def portfolio_benchmark():
    portfolio = portfolio_returns()
    bench = benchmark()

    portfolio['CUM_RET'] = portfolio.groupby(['PORTFOLIO'])['RET'].transform(lambda x: np.cumprod(1 + x))
    v2 = portfolio[portfolio['PORTFOLIO'].str.contains('V2')]
    
    port = v2.groupby('DATE')['CUM_RET'].agg(V2='mean').reset_index()

    port = pd.merge(port, bench, on='DATE', how='left')
    port['RET'] = (port['AVE'] / port['AVE'].shift(1)) -1
    port['BENCH'] = np.cumprod(1 + port['RET'])

    port = port[port['DATE'] >= "2007-01-01"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#1D1D1B')
    ax.set_facecolor('#1D1D1B')

    ax.plot(port['DATE'], port['BENCH'], label='Benchmark', color='red')
    ax.plot(port['DATE'], port['V2'], label='New Portfolio', color='green')

    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Returns", color='white')
    ax.set_title("Total Portfolio vs. Benchmark Cumalitive Return (Growth of $1)", fontsize=12, color='white')
    ax.legend(facecolor='#1D1D1B', edgecolor='white', fontsize=8, labelcolor='white')
    ax.grid(True, linestyle=':', color='white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.show()

from scipy.stats import linregress
from numpy.polynomial import Polynomial

def famaFrench():
    portfolio = portfolio_returns()

    famma = pd.read_csv(r"C:\Users\Brayden Boyko\Downloads\Fixed.csv")
    famma['DATE'] = pd.to_datetime(famma['date'])
    famma['FAMMA'] = (famma['lo20'] + famma['hi20']) / 2

    portfolio = portfolio[portfolio['PORTFOLIO'].isin(['LongPosV1', 'ShortPosV1'])]
    portfolio = portfolio.groupby(['DATE'], as_index=False)['RET'].mean()

    portfolio = pd.merge(portfolio, famma, on='DATE', how='left')
    portfolio = portfolio.dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1D1D1B')
    ax.set_facecolor('#1D1D1B')

    ax.scatter(portfolio['FAMMA'], portfolio['RET'], color='#9fc3cf', alpha=0.6)

    slope, intercept, r_value, _, _ = linregress(portfolio['FAMMA'], portfolio['RET'])
    regression_line = slope * portfolio['FAMMA'] + intercept

    coeffs = Polynomial.fit(portfolio['FAMMA'], portfolio['RET'], 2).convert().coef
    sorted_FAMMA = np.sort(portfolio['FAMMA'])
    polynomial_fit_sorted = coeffs[0] + coeffs[1] * sorted_FAMMA + coeffs[2] * sorted_FAMMA**2

    y_actual = portfolio['RET']
    y_predicted = coeffs[0] + coeffs[1] * portfolio['FAMMA'] + coeffs[2] * (portfolio['FAMMA'] ** 2)
    r_squared = 1 - (((y_actual - y_predicted) ** 2).sum() / ((y_actual - y_actual.mean()) ** 2).sum())

    ax.plot(portfolio['FAMMA'], regression_line, color='red', label=f"Regression (R²={r_value**2:.2f})")

    ax.plot(sorted_FAMMA, polynomial_fit_sorted, color='orange', label=f"Polynomial (R²={r_squared:.2f})")

    ax.set_title("New Portfolio vs. Famma-French IVOL Portfolio", fontsize=12, color='white', loc='left')
    ax.set_xlabel("Famma-French IVOL Portfolio Return", fontsize=10, color='white')
    ax.set_ylabel("New Portfolio Return", fontsize=10, color='white')
    
    ax.tick_params(axis='x', labelsize=8, colors='white')
    ax.tick_params(axis='y', labelsize=8, colors='white')

    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    ax.legend(facecolor='#1D1D1B', edgecolor='white', fontsize=10, labelcolor='white')
    ax.grid(True, linestyle=':', color='white')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.show()
