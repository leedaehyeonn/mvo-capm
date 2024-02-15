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

def risk_return(stocks, start, end):  
    yfin.pdr_override()
    stockdata = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockdata = stockdata['Adj Close']
    returns = stockdata.pct_change()
    mean_returns = returns.mean() * 250
    stdev = returns.std() * np.sqrt(250)

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

# def risk_return_profile():
#     returns = np.sum(mean_returns*weights)*252
#     std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
#     return returns,std

ETF_list = ['ACWI','EFA','EZU','IVV', 'IJR', 'IEFA', 'IEMG','BND', 'BNDX',  'IAU', 'VNQ']
end = dt.datetime.now()
start = end - relativedelta(years=10)

keys = ETF_list
values = ['글로벌 주식', '선진국 주식', '유럽 주식','미국 대형주', '미국 소형주', '선진국 주식(코어)', '신흥국 주식','미국 채권','미국 외 글로벌 채권','금', '부동산' ]
df = dict(zip(keys, values))


data = risk_return(ETF_list, start, end)
risk_return_plot(ETF_list, start, end)

# data = pdr.get_data_yahoo(ETF_list, start, end)['Adj Close']
# mean = data.pct_change().mean()*250
# stdev = data.pct_change().std()*np.sqrt(250)

# np.corrcoef(data['Mean'],data['Mean'].T)

# stats = pd.DataFrame({'Mean': mean, 'Standard Deviation': stdev})
# stats_percent = stats*100

# ax = stats_percent.plot(kind='scatter', x=['Standard Deviation'], y=['Mean'])
# for i, txt in enumerate(stats_percent.index):
#     ax.annotate(df[txt], (stats_percent.iloc[i]['Standard Deviation'], stats_percent.iloc[i]['Mean']))
    
# plt.title('Risk Return profiles of ETFs')
# plt.xlabel('Standard deviation (%)')
# plt.ylabel('Mean return (%)')

# plt.show()




#일별 수익률 상관관계
data = pdr.get_data_yahoo(ETF_list, start, end)['Adj Close']
data = data.pct_change().dropna()
data = data*100
data
result = data.corr()
result = round(result,2)
table = pd.DataFrame(result)
table = table.rename(columns=df)
table = table.set_index(table.columns)
table
df 
result