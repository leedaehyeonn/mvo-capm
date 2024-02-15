
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf 
import matplotlib.pyplot as plt
yf.pdr_override() 
import seaborn as sns
sns.set()
plt.style.use('fivethirtyeight')


def getBetaAlpha(stock, market, start, end):
    stockdata = pd.DataFrame(pdr.get_data_yahoo(stock, start=start, end=end)['Adj Close'].pct_change().dropna()).rename(columns = {'Adj Close':stock})
    marketdata = pd.DataFrame(pdr.get_data_yahoo(market, start=start, end=end)['Adj Close'].pct_change().dropna()).rename(columns = {'Adj Close':market})
    data = pd.concat([stockdata, marketdata], axis=1)
    beta, alpha = np.polyfit(x=data[market], y=data[stock],deg=1)
    return beta, alpha

def CAPM_return(stock,market,start,end):
    expected_market_return = pd.DataFrame(pdr.get_data_yahoo(market, start=start, end=end)['Adj Close'].pct_change().dropna()).mean()
    return (getBetaAlpha(stock,market,start,end)[0]*expected_market_return[0])*250

def SML(stock, market, start, end):
    stockdata = pd.DataFrame(pdr.get_data_yahoo(stock, start=start, end=end)['Adj Close'].pct_change().dropna()).rename(columns = {'Adj Close':stock})
    marketdata = pd.DataFrame(pdr.get_data_yahoo(market, start=start, end=end)['Adj Close'].pct_change().dropna()).rename(columns = {'Adj Close':market})
    data = pd.concat([stockdata, marketdata], axis=1)
    beta, alpha = np.polyfit(x=data[market], y=data[stock],deg=1)
    market_return = data[market].mean()
    CAPM = beta*(market_return)
    HistoricalReturn = data[stock].mean()
    return beta, CAPM, HistoricalReturn

def plot(stocklist, market, start, end):
    beta, capm, historicalreturn = SML(stocklist, market, start, end)
    alpha = getBetaAlpha(stocklist, market, start, end)[1]
    df = {}
    df['stocklist'] = stocklist
    df['beta'] = beta 
    df['capm'] = capm * 250 * 100 #annualized & percentage
    df['Historical Return(%)'] = historicalreturn * 250 * 100 #annualized & percentage
    df = pd.DataFrame(df)
    
    plt.figure(figsize = (13,9))
    plt.axvline(0, color='grey', alpha = 0.5)
    plt.axhline(0, color='grey', alpha = 0.5)

    sns.scatterplot(y = 'Historical Return(%)', x = 'beta', data = df, label = 'Historical Returns')
    sns.lineplot(x = beta, y = df['capm'], color = 'red', label = 'CAPM Line')

    return plt.show()



"""SETTING"""
stocklist =['AAPL','TSLA','MSFT'] #,'TSLA','MS','MSFT'
market = '^GSPC'
start = pd.to_datetime('2019-01-01') 
end = pd.to_datetime('2024-01-01')
""""""""""""

plot(stocklist, market, start, end)


