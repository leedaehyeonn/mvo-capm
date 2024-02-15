
import pandas as pd
import numpy as np
from pykrx import stock
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('fivethirtyeight')

def get_multiple_stock_prices(tickers, start_date, end_date):
    stock_prices = {}
    for ticker in tickers:
        df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker, adjusted= True)
        stock_prices[ticker] = df['종가']  # '종가' 컬럼만 선택하여 딕셔너리에 추가
    return pd.DataFrame(stock_prices)

def getBetaAlpha(targetlist, fromdate, todate, market='1001'):
    stockdata = pd.DataFrame(get_multiple_stock_prices(targetlist, fromdate, todate).pct_change())
    marketdata = pd.DataFrame(stock.get_index_ohlcv(fromdate, todate, market)['종가'].pct_change()).rename(columns={'종가':'Market'})
    data = pd.concat([stockdata, marketdata], axis=1)
    data = data.replace(np.nan,0)
    beta, alpha = np.polyfit(x=data['Market'], y=data[targetlist],deg=1) #alpha = Rf(1-beta)... historical return = market_return*beta + alpha
    return beta, alpha

def SML(targetlist, fromdate, todate, market='1001'):
    stockdata = pd.DataFrame(get_multiple_stock_prices(targetlist, fromdate, todate).pct_change())
    marketdata = pd.DataFrame(stock.get_index_ohlcv(fromdate, todate, market)['종가'].pct_change()).rename(columns={'종가':'Market'})
    data = pd.concat([stockdata, marketdata], axis=1)
    data = data.replace(np.nan,0)
    beta, alpha = np.polyfit(x=data['Market'], y=data[targetlist],deg=1)

    market_return = data['Market'].mean()
    CAPM = beta*(market_return)
    Historicalreturn = np.array(data[targetlist].mean())
    return beta, CAPM, Historicalreturn

# def CAPM_(targetlist, fromdate, todate, market='1001',riskfree=0):
#     stockdata = pd.DataFrame(get_multiple_stock_prices(targetlist, fromdate, todate, adjusted=True).pct_change())
#     marketdata = pd.DataFrame(stock.get_index_ohlcv(fromdate, todate, market, adjusted=True)['종가'].pct_change()).rename(columns={'종가':'Market'})
#     data = pd.concat([stockdata, marketdata], axis=1)
#     data = data.replace(np.nan,0)
#     beta, alpha = np.polyfit(x=data['Market']-riskfree, y=data[targetlist]-riskfree,deg=1)
#     market_return = data['Market'].mean()
#     CAPM = riskfree+beta*(market_return-riskfree)
#     Historicalreturn = np.array(data[targetlist].mean())
#     return beta, CAPM, Historicalreturn

def getdataframe(targetlist, start, end, market='1001'):
    beta, capm, historicalreturn = SML(targetlist, start, end, market)
    df = {}
    df['stocklist'] = targetlist
    df['beta'] = beta 
    df['capm(annuallized %)'] = capm * 250 * 100
    df['Historical Return(annualized %)'] = historicalreturn * 250 * 100
    df = pd.DataFrame(df)
    stockname = [stock.get_market_ticker_name(i) for i in targetlist]
    df['stocklist'] = stockname
    return df

def plot(targetlist, start, end, market='1001'):
    df = getdataframe(targetlist, start, end, market)
    
    plt.figure(figsize = (13,9))

    plt.axvline(0, color='grey', alpha = 0.5)
    plt.axhline(0, color='grey', alpha = 0.5)

    sns.scatterplot(y = 'Historical Return(annualized %)', x = 'beta', data = df, label = 'Historical Returns')
    sns.lineplot(y = df['capm(annuallized %)'], x = df['beta'], color = 'red', label = 'CAPM Line')

    return plt.show()

"""""""""""""""""""""""""SETTING"""""""""""""""""""""""""
stocklist = stock.get_market_ticker_list("20240101", 'ALL')
stockname = [stock.get_market_ticker_name(i) for i in stocklist]
targetlist = random.sample(stocklist, 5)
fromdate = "20200101"
todate = '20240101'

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
plot(targetlist, fromdate, todate, market='1001')
get_multiple_stock_prices(targetlist,fromdate,todate)
getdataframe(targetlist,fromdate,todate)

""""""""""""""""""""""""""""""""""""""""""""""""""

