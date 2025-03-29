import pandas as pd
import numpy as np
from scipy.stats import linregress

from monthly_returns import portfolio_returns

def ivol_ret():

    ivol = portfolio_returns()

    ivol = ivol[ivol['PORTFOLIO'].isin(['LongPosV1', 'ShortPosV1'])]

    ivol_v = ivol.pivot(index='DATE', columns='PORTFOLIO', values='IVOL').reset_index()
    print(ivol_v)
    ivol_v['IVOL'] = (ivol_v['LongPosV1'] + ivol_v['ShortPosV1']) / 2
    ivol_v['IVOL'] = ivol_v['IVOL'] - ivol_v['IVOL'].shift(1)
    ivol_v = ivol_v[['DATE', 'IVOL']]

    ivol_r = ivol.pivot(index='DATE', columns='PORTFOLIO', values='RET').reset_index()
    ivol_r['RET'] = (ivol_r['LongPosV1'] + ivol_r['ShortPosV1']) / 2
    ivol_r = ivol_r[['DATE', 'RET']]

    df = pd.merge(ivol_v, ivol_r, on='DATE', how='left')
    df = df[df['DATE'] >= "2007-01-01"]

    results = linregress(df['IVOL'],df['RET'])

    cor = np.corrcoef(df['IVOL'],df['RET'])

    print(results, cor)

ivol_ret()
