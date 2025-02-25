import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

from screens import v1, v2, rebalance

def data():
    dataFile = r"C:\Users\Brayden Boyko\Downloads\bizbhzwdqwkigpgu.csv"
    df = pd.read_csv(dataFile)

    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
    df = df.sort_values(by='DATE', ascending=True)
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year

    def convert_to_numeric(column):
        return pd.to_numeric(column.str.replace('%', ''), errors='coerce')
    
    df['RET'] = convert_to_numeric(df['RET']) / 100
    df['IVOL'] = convert_to_numeric(df['ivol']) / 100

    df = df[['DATE', 'MONTH', 'YEAR', 'TICKER', 'RET', 'IVOL']]

    return df

def metaData(select=None):
    df = data()
    if select == 'tickerCount':
        ticker_count = df['TICKER'].nunique()
        return ticker_count
    
    elif select == 'tickers':
        tickers = list(df['TICKER'].unique())
        return tickers
    
    elif select == 'years':
        years = list(df['YEAR'].unique())
        return years

def portfolio():
    df = data()
    v1_list = [] ## Cols: Year, LongPosV1, ShortPosV1, LongRetV1, ShortRetV1, LongSkewV1, ShortSkewV1
    
    rebalance_period = metaData(select='years')

    ## V1->V2 Screen (Extreme IVOL High-Low) -> (Extreme SKEW Positive-Negative)
    for period in rebalance_period:
        df_back = df[(df['YEAR'] <= (period - 1)) & (df['YEAR'] >= (period - 6))].copy()  ## Five Year Lookback Window
        df_back['PERIOD'] = period ## Rebalance Year
        
        stats_df = df_back.groupby('TICKER')['IVOL'].agg(
            AVE='mean',
            SKEW='skew'
            ).reset_index()
        
        return_df = df[df['YEAR'] == period]
        return_df = df_back.groupby('TICKER')['RET'].agg(
            T_RETURN='sum'
            ).reset_index()
        
        df_back = df_back.merge(stats_df, on='TICKER', how='left')
        df_back = df_back.merge(return_df, on='TICKER', how='left')
        df_back = df_back.drop_duplicates(subset='TICKER')
        df_back = df_back.dropna()
        df_back = df_back.sort_values(by='AVE', ascending=True)

        long = df_back.head(100).copy()
        short = df_back.tail(100).copy()

        long['Position'] = 'Long'
        short['Position'] = 'Short'

        combined = pd.concat([long, short], axis=0)

        combined['Year'] = period
        combined = combined[['Year', 'TICKER', 'Position', 'T_RETURN', 'SKEW']]

        v1_list.append(combined)

    v1_df = pd.concat(v1_list, ignore_index=True)
    v1_df = v1_df.pivot(index=['Year', 'TICKER'], columns='Position', values=['T_RETURN', 'SKEW']).reset_index()
    v1_df.columns = ['Year', 'TICKER', 'LongRetV1', 'ShortRetV1', 'LongSkewV1', 'ShortSkewV1']
    v1_df = v1_df[v1_df['Year'] >= 2007] ## Rebalances Begin In 2007
    
    print(v1_df)






portfolio()


