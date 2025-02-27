from monthly_returns import portfolio_returns

import pandas as pd
import matplotlib.pyplot as plt

def monthly_ret():
    portfolio = portfolio_returns()

    plt.figure(figsize=(12, 6))
    for port in portfolio['PORTFOLIO'].unique():
        data = portfolio[portfolio['PORTFOLIO'] == port]
        plt.plot(data['DATE'], data['RET'], label=port)

    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.title("Monthly Portfolio Returns Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def monthly_ivol():
    portfolio = portfolio_returns()

    plt.figure(figsize=(12, 6))
    for port in portfolio['PORTFOLIO'].unique():
        data = portfolio[portfolio['PORTFOLIO'] == port]
        plt.plot(data['DATE'], data['IVOL'], label=port)

    plt.xlabel("Date")
    plt.ylabel("IVOL")
    plt.title("Monthly Portfolio Idiosyncratic Volatility Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

monthly_ivol()