import pandas as pd
import numpy as np

def data():
    dataFile = r"C:\Users\Brayden Boyko\Downloads\bizbhzwdqwkigpgu.csv"
    dataFile2 = r"C:\Users\brayd\Downloads\bizbhzwdqwkigpgu.csv"
    dataFile3 = r"C:\Users\brayd\Downloads\kem6kumzrbylvuz3.csv"
    df = pd.read_csv(dataFile3)

    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
    df = df.sort_values(by='DATE', ascending=True)
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year
    df['YEAR_MONTH'] = df['DATE'].dt.to_period('M').dt.to_timestamp()

    def convert_to_numeric(column):
        return pd.to_numeric(column.str.replace('%', ''), errors='coerce')
    
    df['RET'] = convert_to_numeric(df['RET']) / 100
    df['IVOL'] = convert_to_numeric(df['ivol']) / 100

    df = df[['DATE','YEAR_MONTH','TICKER', 'RET', 'IVOL']]

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
        years = list(df['YEAR_MONTH'].unique())
        return years


def portfolio():
    df = data()
    v1_list = [] ## Cols: Year, LongPosV1, ShortPosV1, LongRetV1, ShortRetV1, LongSkewV1, ShortSkewV1
    
    rebalance_period = metaData(select='years')

    ### + Add Random Sample

    ## V1->V2 Screen (Extreme IVOL High-Low) -> (Extreme SKEW Positive-Negative)
    for period in rebalance_period:
        period_timestamp = pd.to_datetime(period)
        df_back = df[(df['YEAR_MONTH'] <= period_timestamp - pd.DateOffset(months=1)) & 
             (df['YEAR_MONTH'] >= period_timestamp - pd.DateOffset(months=37))].copy()
        df_back['PERIOD'] = period ## Rebalance Year
        
        stats_df = df_back.groupby('TICKER')['IVOL'].agg(
            AVE='mean',
            ).reset_index()
        
        stats2_df = df_back.groupby('TICKER').agg(
            SKEW=('RET','skew'),
            RET_M =('RET', 'mean'),
            AVE_I=('IVOL', 'mean')
        ).reset_index()

        stats2_df['SKEW'] = stats2_df['SKEW_ADJ'] = stats2_df['SKEW'] * stats2_df['RET_M'].apply(lambda x: -abs(x)) / stats2_df['AVE_I'] ## THIS IS OUR FILTER
        
        return_df = df[df['YEAR_MONTH'] == period]
        return_df = df_back.groupby('TICKER')['RET'].apply(lambda x: (x + 1).prod() - 1).reset_index(name='T_RETURN')

        
        df_back = df_back.merge(stats_df, on='TICKER', how='left')
        df_back = df_back.merge(stats2_df, on='TICKER', how='left')
        df_back = df_back.merge(return_df, on='TICKER', how='left')
        df_back = df_back.drop_duplicates(subset='TICKER')
        df_back = df_back.dropna()
        df_back = df_back.sort_values(by='AVE', ascending=True)

        long = df_back.head(100).copy()
        short = df_back.tail(100).copy()

        long['Position'] = 'Long'
        short['Position'] = 'Short'

        combined = pd.concat([long, short], axis=0)

        combined['YEAR_MONTH'] = period
        combined = combined[['YEAR_MONTH', 'TICKER', 'Position', 'T_RETURN', 'SKEW']]

        v1_list.append(combined)

    v1_df = pd.concat(v1_list, ignore_index=True)
    v1_df = v1_df.pivot(index=['YEAR_MONTH', 'TICKER'], columns='Position', values=['T_RETURN', 'SKEW']).reset_index()
    v1_df.columns = ['YEAR_MONTH', 'TICKER', 'LongRetV1', 'ShortRetV1', 'LongSkewV1', 'ShortSkewV1']
    v1_df = v1_df[v1_df['YEAR_MONTH'] >= "2007-01"] ## Rebalances Begin In 2007
    rebalance_years = list(v1_df['YEAR_MONTH'].unique())

    v2_list = []
    for year in rebalance_years:
        v2_df = v1_df[v1_df['YEAR_MONTH'] == year]
        
        v2_long = v2_df[['YEAR_MONTH', 'TICKER','LongRetV1', 'LongSkewV1']].dropna()
        v2_long.columns = ['YEAR_MONTH', 'TICKER', 'LongRetV2', 'LongSkewV2']
        v2_long = v2_long.sort_values(by='LongSkewV2', ascending=True)
        v2_long = v2_long.nlargest(30, 'LongSkewV2') ## OG STRATEGY: v2_long.tail(30)

        v2_short = v2_df[['YEAR_MONTH', 'TICKER','ShortRetV1', 'ShortSkewV1']].dropna()
        v2_short.columns = ['YEAR_MONTH', 'TICKER', 'ShortRetV2', 'ShortSkewV2']
        v2_short = v2_short.sort_values(by='ShortSkewV2', ascending=True)
        v2_short = v2_short.nsmallest(30, 'ShortSkewV2') ## OG STRATEGY: v2_long.head(30)

        combined = pd.concat([v2_long, v2_short], axis=0)
        v2_list.append(combined)

    v2_df = pd.concat(v2_list, ignore_index=True)

    ## Long-Short Base Startegy (L_Base/S_Base) ---> Long-Short Portfolio Startegy (L_Port/S_Port)

    L_Base = v1_df[['YEAR_MONTH', 'TICKER', 'LongRetV1', 'LongSkewV1']].dropna()
    tickers = L_Base.groupby('YEAR_MONTH')['TICKER'].unique().reset_index(name='LongPosV1')
    L_Base = L_Base.groupby('YEAR_MONTH').agg(LongV1Ret=('LongRetV1', 'mean'), LongV1Skew=('LongSkewV1', 'mean')).reset_index()
    L_Base = L_Base.merge(tickers, on='YEAR_MONTH', how='left')

    S_Base = v1_df[['YEAR_MONTH', 'TICKER', 'ShortRetV1', 'ShortSkewV1']].dropna()
    tickers = S_Base.groupby('YEAR_MONTH')['TICKER'].unique().reset_index(name='ShortPosV1')
    S_Base = S_Base.groupby('YEAR_MONTH').agg(ShortV1Ret=('ShortRetV1', 'mean'), ShortV1Skew=('ShortSkewV1', 'mean')).reset_index()
    S_Base['ShortV1Ret'] = (S_Base['ShortV1Ret'] * -1).clip(lower=-1)  # Make Returns Inverse
    S_Base = S_Base.merge(tickers, on='YEAR_MONTH', how='left')

    L_Port = v2_df[['YEAR_MONTH', 'TICKER', 'LongRetV2', 'LongSkewV2']].dropna()
    tickers = L_Port.groupby('YEAR_MONTH')['TICKER'].unique().reset_index(name='LongPosV2')
    L_Port = L_Port.groupby('YEAR_MONTH').agg(LongV2Ret=('LongRetV2', 'mean'), LongV2Skew=('LongSkewV2', 'mean')).reset_index()
    L_Port = L_Port.merge(tickers, on='YEAR_MONTH', how='left')

    S_Port = v2_df[['YEAR_MONTH', 'TICKER', 'ShortRetV2', 'ShortSkewV2']].dropna()
    tickers = S_Port.groupby('YEAR_MONTH')['TICKER'].unique().reset_index(name='ShortPosV2')
    S_Port = S_Port.groupby('YEAR_MONTH').agg(ShortV2Ret=('ShortRetV2', 'mean'), ShortV2Skew=('ShortSkewV2', 'mean')).reset_index()
    S_Port['ShortV2Ret'] = (S_Port['ShortV2Ret'] * -1).clip(lower=-1)  # Make Returns Inverse
    S_Port = S_Port.merge(tickers, on='YEAR_MONTH', how='left')

    port = L_Base.merge(L_Port, how='left', on='YEAR_MONTH')
    port = port.merge(S_Base, how='left', on='YEAR_MONTH')
    port = port.merge(S_Port, how='left', on='YEAR_MONTH')

    return port

def pchart():
    return portfolio()

def monthly_returns(Pos=None, Port=None):
    """
    ## Generate a DataFrame With Monthly Returns For Specified Portfolio 
    ---
    + Pos: Position Argument ("Long", "Short")
    + Port: Portfolio ("V1", "V2")
    """

    df = data()
    port = portfolio()
    select = f"{Pos}Pos{Port}"
    skew = f"{Pos}{Port}Skew"
    
    port = port[['YEAR_MONTH', f'{select}', f'{skew}']]

    df = df.merge(port, how='left', left_on='YEAR_MONTH', right_on='YEAR_MONTH').dropna()
    df = df.explode(f'{select}')
    df = df[df['TICKER'] == df[f'{select}']]
    df['PORTFOLIO'] = select

    ## Converts SKEW Column Headers
    possible_names = {"ShortV2Skew", "ShortV1Skew", "LongV2Skew", "LongV1Skew"}
    existing_name = next((col for col in df.columns if col in possible_names), None)
    if existing_name:
        df = df.rename(columns={existing_name: "SKEW"})

    df = df[['DATE', 'YEAR_MONTH', 'TICKER', 'RET', 'IVOL', 'SKEW', 'PORTFOLIO']]

    return df

def portfolio_returns():

    L_v1 = monthly_returns(Pos="Long", Port="V1")
    S_v1 = monthly_returns(Pos="Short", Port="V1")
    L_v2 = monthly_returns(Pos="Long", Port="V2")
    S_v2 = monthly_returns(Pos="Short", Port="V2")

    S_v1['RET'] = S_v1['RET'] * -1
    S_v2['RET'] = S_v2['RET'] * -1

    portfolio = pd.concat([L_v1, S_v1, L_v2, S_v2])
    portfolio = portfolio.groupby(['DATE', 'PORTFOLIO']).agg({'RET': 'mean', 'IVOL': 'mean', 'SKEW': 'mean'}).reset_index()

    return portfolio

import matplotlib.pyplot as plt
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


