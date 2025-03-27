import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pandas.tseries.offsets import MonthBegin

from monthly_returns import portfolio_returns

esg = pd.read_csv(r"C:\Users\brayd\Downloads\multiTimeline (3).csv")
csr = pd.read_csv(r"C:\Users\brayd\Downloads\multiTimeline (4).csv")
ivol = portfolio_returns()
ivol = ivol[ivol['PORTFOLIO'].isin(['LongPosV1', 'ShortPosV1'])]
ivol = ivol.pivot(index='DATE', columns='PORTFOLIO', values='IVOL').reset_index()
ivol['Value'] = (ivol['LongPosV1'] + ivol['ShortPosV1']) / 2
ivol['DATE'] = pd.to_datetime(ivol['DATE']) - MonthBegin(1)
ivol = ivol[['DATE', 'Value']]


def ESG():
    df = esg
    df.columns = ['Month', 'Value']

    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values(by=['Month'])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#1D1D1B')
    ax.set_facecolor('#1D1D1B')

    ax.plot(df['Month'], df['Value'], label='ESG Search', color='#9fc3cf')

    ax.tick_params(axis='x', rotation=45) 

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ax.set_xlabel("Month", fontsize=10, color='white')
    ax.set_ylabel("Indexed Search", fontsize=10, color='white')
    ax.set_title("Search Indexed Frequency For Term: ESG In The United States (Google Trends)",
                 fontsize=12, color='white', loc='left')
    ax.grid(True, linestyle=':', color='white')

    plt.tight_layout()
    plt.show()

def CSR():
    df = csr
    df.columns = ['Month', 'Value']

    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values(by=['Month'])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#1D1D1B')
    ax.set_facecolor('#1D1D1B')

    ax.plot(df['Month'], df['Value'], label='CSR Search', color='#9fc3cf')

    ax.tick_params(axis='x', rotation=45) 

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ax.set_xlabel("Month", fontsize=10, color='white')
    ax.set_ylabel("Indexed Search", fontsize=10, color='white')
    ax.set_title("Search Indexed Frequency For Term: CSR In The United States (Google Trends)",
                 fontsize=12, color='white', loc='left')
    ax.grid(True, linestyle=':', color='white')

    plt.tight_layout()
    plt.show()

def INTEREST():
    df = pd.merge(esg, csr, on='Month', how='left')

    df['Value'] = df['Value_x'] + df['Value_y']
    df['DATE'] = pd.to_datetime(df['Month'])
    df = df[['DATE', 'Value']]

    df = pd.merge(df, ivol, on='DATE', how='left')
    df = df.sort_values(by=['DATE'])
    df = df[df['DATE'] <= '2021-12-01']
    df = df.dropna()

    df['search'] = df['Value_x'] / df['Value_x'].iloc[0]
    df['ivol'] = df['Value_y'] / df['Value_y'].iloc[0]

    corr = np.corrcoef(df['search'], df['ivol'])

    print(corr)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#1D1D1B')
    ax.set_facecolor('#1D1D1B')

    ax.plot(df['DATE'], df['search'], label='Total Search', color='#4A90E2')
    ax.plot(df['DATE'], df['ivol'], label='IVOL', color='#9fc3cf')

    ax.tick_params(axis='x', rotation=45) 

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ax.set_xlabel("Month", fontsize=10, color='white')
    ax.set_ylabel("Indexed Value (Scale: 1)", fontsize=10, color='white')
    ax.set_title("CSR/ESG Search Compared To IVOL (Indexed Values of Scale 1)",
                 fontsize=12, color='white', loc='left')
    ax.grid(True, linestyle=':', color='white')

    plt.tight_layout()
    plt.legend(facecolor='#1D1D1B',
               edgecolor='white',
               fontsize=8,
               loc='upper left',
               title='',
               title_fontsize=10,
               labelcolor='white'
               )
    plt.show()



INTEREST()
