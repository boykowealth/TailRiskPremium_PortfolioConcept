from portfolio import portfolio, data

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

    print(select)
    
    port = port[[]]



    print(portfolio)


monthly_returns(Pos="Long", Port="V1")