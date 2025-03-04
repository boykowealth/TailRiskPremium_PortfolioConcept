from portfolio import pchart

import pandas as pd

def tickers():

    data = pchart()

    data = data[['Year', 'LongPosV2', 'ShortPosV2']]
    exploded_long = data.explode('LongPosV2').rename(columns={'LongPosV2': 'Ticker'}).assign(Portfolio='Long')
    exploded_short = data.explode('ShortPosV2').rename(columns={'ShortPosV2': 'Ticker'}).assign(Portfolio='Short')

    data = pd.concat([exploded_long, exploded_short], ignore_index=True)
    data = data[['Year', 'Portfolio', 'Ticker']]

    return data

tickers()