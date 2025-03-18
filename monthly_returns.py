from portfolio import portfolio, data
import pandas as pd

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
    
    port = port[['Year', f'{select}', f'{skew}']]

    df = df.merge(port, how='left', left_on='YEAR', right_on='Year').dropna()
    df = df.explode(f'{select}')
    df = df[df['TICKER'] == df[f'{select}']]
    df['PORTFOLIO'] = select

    ## Converts SKEW Column Headers
    possible_names = {"ShortV2Skew", "ShortV1Skew", "LongV2Skew", "LongV1Skew"}
    existing_name = next((col for col in df.columns if col in possible_names), None)
    if existing_name:
        df = df.rename(columns={existing_name: "SKEW"})

    df = df[['DATE', 'YEAR', 'TICKER', 'RET', 'IVOL', 'SKEW', 'PORTFOLIO']]

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