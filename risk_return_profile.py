import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
from pykrx import stock
import yfinance as yfin
import scipy.optimize as sc
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'  # NanumGothic 폰트를 사용하도록 설정

def getdata(stocks, start, end):
    yfin.pdr_override()
    stockdata = pdr.get_data_yahoo(ETF_list, start=start, end=end)
    stockdata = stockdata['Adj Close'].resample('M').last().pct_change().dropna()
    return stockdata

def risk_return(stocks, start, end):  
    yfin.pdr_override()
    stockdata = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockdata = stockdata['Adj Close']
    returns = stockdata.pct_change()
    mean_returns = returns.mean() * 250
    stdev = returns.std() * np.sqrt(250)

    stats = pd.DataFrame({'Mean': mean_returns, 'Standard Deviation': stdev})
    return stats

def risk_return_monthly(stocks, start, end):  
    yfin.pdr_override()
    stockdata = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockdata = stockdata['Adj Close']
    
    # 월별 데이터로 리샘플링
    monthly_stockdata = stockdata.resample('M').last()
    returns = monthly_stockdata.pct_change()
    mean_returns = returns.mean() * 12  
    stdev = returns.std() * np.sqrt(12) 

    stats = pd.DataFrame({'Mean': mean_returns, 'Standard Deviation': stdev})
    return stats

def risk_return_plot(stocks, start, end):
    stats = risk_return(stocks, start, end)
    stats_percent = stats*100
    ax = stats_percent.plot(kind='scatter', x=['Standard Deviation'], y=['Mean'])
    for i, txt in enumerate(stats_percent.index):
        ax.annotate(df[txt], (stats_percent.iloc[i]['Standard Deviation'], stats_percent.iloc[i]['Mean']))
        
    plt.title('Risk Return profiles of ETFs')
    plt.xlabel('Standard deviation (%)')
    plt.ylabel('Mean return (%)')

    return plt.show()

def correlation(ETF_list, start, end):
    data = pdr.get_data_yahoo(ETF_list, start, end)['Adj Close']
    monthly_stockdata = data.resample('M').last()
    returns = monthly_stockdata.pct_change()*100
    result = returns.corr()
    result = round(result,2)
    table = pd.DataFrame(result)
    table = table.rename(columns=df)
    table = table.set_index(table.columns)
    return table

"""""""""""""""""Setting"""""""""""""""""""""
# ETF_list = ['ACWI','EFA','EZU','IVV', 'IJR', 'IEFA', 'IEMG','BND', 'BNDX',  'IAU', 'VNQ']
ETF_list = ['ACWI','BND', 'HYG',  'IAU', 'VNQ']
end = dt.datetime.now()
start = end - relativedelta(years=15)

keys = ETF_list
# values = ['글로벌 주식', '선진국 주식', '유럽 주식','미국 대형주', '미국 소형주', '선진국 주식(코어)', '신흥국 주식','미국 채권','미국 외 글로벌 채권','금', '부동산' ]
values = ['글로벌 주식', '미국 채권','하이일드','금', '부동산' ]
df = dict(zip(keys, values))
"""""""""""""""""""Setting"""""""""""""""""""""

data = risk_return(ETF_list, start, end)
risk_return_plot(ETF_list, start, end)
data2 = risk_return_monthly(ETF_list, start, end)
data2
correlation(ETF_list,start,end)


data3 = getdata(ETF_list,start,end)
data3 = data3.rename(columns=df) 
data3



#######################
########plot1###########
#######################
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[2,3].set_visible(False)     
subplot_index = 1
for i in range(2):
    for j in range(3):
        if subplot_index <= len(data3.columns):
            column_name = data3.columns[subplot_index - 1]
            axs[i, j].hist(data3[column_name], bins = 100, label=column_name)
            axs[i, j].set_title(column_name)
            subplot_index += 1
   
plt.show()
#######################
########plot2###########
#######################

fig, ax = plt.subplots(figsize=(10, 6))

for column in data3.columns:
    ax.plot(data3[column], alpha=0.5, label=column)

ax.set_title('Asset Returns')
ax.set_xlabel('Returns')
ax.set_ylabel('Frequency')
ax.legend()

plt.show()

#######################
########plot2.2###########
#######################
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = go.Figure()

for column in data3.columns:
    fig.add_trace(go.Scatter(x = data3.index, y = data3[column], name = column))

fig.update_layout(title_text='Asset Returns', title_x=0.5)
fig.show()
