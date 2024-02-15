import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
from pykrx import stock
import yfinance as yfin
import scipy.optimize as sc
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

#Import data
def getData(stocks, start, end):  
    yfin.pdr_override()
    stockdata = pdr.get_data_yahoo(stocks, start=start, end=end, interval='1mo')
    stockdata = stockdata['Adj Close']
    returns = stockdata.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

def portfolioperformance(weights, mean_returns, cov_matrix):
    "Annualize"
    returns = np.sum(mean_returns*weights)*12
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)
    return returns,std

def negative_SR(weights, mean_returns, cov_matrix, riskfree = 0):
    p_returns, p_std = portfolioperformance(weights, mean_returns, cov_matrix)
    return -(p_returns - riskfree)/p_std

def max_SR(mean_returns, cov_matrix, riskfree = 0, constraint_set=(0,1)):
    "Minimize the negative sharpe ratio, by altering weights of the portfolio"
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, riskfree)
    constraints = ({'type':'eq', 'fun': lambda x : np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    result = sc.minimize(
        negative_SR, num_assets*[1./num_assets], args = args,
        method = 'SLSQP', bounds = bounds, constraints = constraints
    )
    return result

def portfoliostd(weights, mean_returns, cov_matrix):
    return portfolioperformance(weights, mean_returns, cov_matrix)[1]

def portfolioreturn(weights, mean_returns, cov_matrix):
    return portfolioperformance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix, constraint_set=(0,1)):
    "Minimize the portfolio variance, by altering weights of aseets in the portfolio"
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type':'eq', 'fun': lambda x : np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    result = sc.minimize(
        portfoliostd, num_assets*[1./num_assets], args = args,
        method = 'SLSQP', bounds = bounds, constraints = constraints
    )
    return result

def calculated_results(mean_returns, cov_matrix, riskfree=0, constraint_set=(0,1)):
    "Read in mean, covmatrix, and other financial information.. Output max sharpe_ratio, min_volatility, NO ANNUALIZED"
    # Max Sharpe ratio portfolio
    max_SR_Portfolio = max_SR(mean_returns, cov_matrix, riskfree, constraint_set)
    max_SR_returns, max_SR_std = portfolioperformance(max_SR_Portfolio['x'], mean_returns, cov_matrix)
    max_SR_allocation = pd.DataFrame(max_SR_Portfolio['x'], index= mean_returns.index, columns=['allocation'])
  
    # Min variance portfolio
    min_VOL_Portfolio = min_variance(mean_returns, cov_matrix)
    min_VOL_returns, min_VOL_std = portfolioperformance(min_VOL_Portfolio['x'], mean_returns, cov_matrix)
    min_VOL_allocation = pd.DataFrame(min_VOL_Portfolio['x'], index= mean_returns.index, columns=['allocation'])

    max_SR_allocation.allocation = [round(i*100,0) for i in max_SR_allocation.allocation]
    min_VOL_allocation.allocation = [round(i*100,0) for i in min_VOL_allocation.allocation]
    # max_SR_returns, max_SR_std =  round(max_SR_returns*100,4), round(max_SR_std*100,4)
    # min_VOL_returns, min_VOL_std =  round(min_VOL_returns*100,4), round(min_VOL_std*100,4)

    return max_SR_returns, max_SR_std, max_SR_allocation, min_VOL_returns, min_VOL_std, min_VOL_allocation

def calculated_table(mean_returns, cov_matrix, riskfree=0, constraint_set=(0,1)):
    max_SR_returns, max_SR_std, max_SR_allocation, min_VOL_returns, min_VOL_std, min_VOL_allocation = calculated_results(mean_returns, cov_matrix, riskfree, constraint_set)
    max_SR_returns, max_SR_std =  round(max_SR_returns*100,4), round(max_SR_std*100,4)
    min_VOL_returns, min_VOL_std =  round(min_VOL_returns*100,4), round(min_VOL_std*100,4)
    allocation_index = max_SR_allocation.index.tolist()
    results = pd.DataFrame({
    'Portfolio': ['Max SR', 'Min Volatility'],
    'Returns (%)': [max_SR_returns, min_VOL_returns],
    'Standard Deviation (%)': [max_SR_std, min_VOL_std],
    'Allocation (%)': [max_SR_allocation.values.flatten(), min_VOL_allocation.values.flatten()],
    'Allocation Index': [allocation_index, allocation_index]})
    
    return results

def efficient_frontier_input(mean_returns, cov_matrix, riskfree=0, constraint_set=(0,1)):
    max_SR_returns, max_SR_std, max_SR_allocation, min_VOL_returns, min_VOL_std, min_VOL_allocation = calculated_results(mean_returns, cov_matrix, riskfree, constraint_set)
    std_range = []
    targetreturns = np.linspace(min_VOL_returns, max_SR_returns+0.01, 20)
    for target_return in targetreturns:
        std_range.append((efficient_frontier(mean_returns, cov_matrix, target_return)['fun'])) #Annualize\
    return std_range, targetreturns


def efficient_frontier(mean_returns, cov_matrix, target_return, constraint_set=(0,1)):
    'For each target_return, optimize the portfolio for min variance'
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = (
        {'type':'eq', 'fun': lambda x: portfolioreturn(x, mean_returns, cov_matrix) - target_return},
        {'type':'eq', 'fun': lambda x : np.sum(x) - 1})
    bound  = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    efficient = sc.minimize(
        portfoliostd, num_assets*[1./num_assets], args = args,
        method = 'SLSQP', bounds = bounds, constraints = constraints
    )
    return efficient

def EF_graph(mean_returns, cov_matrix, riskfree = 0, constraint_set=(0,1)):
    'Return a graph ploting the Min vol, Max sr and efficeint frontier'
    max_SR_returns, max_SR_std, max_SR_allocation, min_VOL_returns, min_VOL_std, min_VOL_allocation = calculated_results(mean_returns, cov_matrix, riskfree, constraint_set)
    std_range, targetreturns = efficient_frontier_input(mean_returns, cov_matrix, riskfree, constraint_set)

    # Max SR
    Max_Sharpe = go.Scatter(
        name = 'Maximum Sharpe Ratio',
        mode = 'markers',
        x = [round(max_SR_std*100,4)],
        y = [round(max_SR_returns*100,4)],
        marker = dict(color = 'red', size = 14, line = dict(width = 3, color = 'black'))
    )

    # Min Vol
    Min_Vol = go.Scatter(
        name = 'Minimum Volatility',
        mode = 'markers',
        x = [round(min_VOL_std*100,4)],
        y = [round(min_VOL_returns*100,4)],
        marker = dict(color = 'green', size = 14, line = dict(width = 3, color = 'black'))
    )

   # Efficient Frontier
    EF_curve = go.Scatter(
        name = 'Efficient Frontier',
        mode = 'lines',
        x = [round(std_range*100,4) for std_range in std_range],
        y = [round(targetreturns*100,4) for targetreturns in targetreturns],
        line = dict(color = 'black', width = 4, dash = 'solid')
    )

    # # Capital Market Line (CML)
    # x_cml = np.linspace(0, max_SR_std*100, 100)  # 표준편차 범위
    # y_cml = riskfree*100 + ((max_SR_returns - riskfree)/ (max_SR_std))*x_cml
    # CML_line = go.Scatter(
    #     name = 'Capital Market Line',
    #     mode = 'lines',
    #     x = x_cml,
    #     y = y_cml,
    #     line = dict(color = 'blue', width = 2, dash = 'dash')
    # )
    
    data = [Max_Sharpe, Min_Vol, EF_curve]
    layout = go.Layout(
        title ='Portfolio Optimization with the efficient frontier',
        yaxis = dict(title='Annualized Returns (%)'),
        xaxis = dict(title='Annualized Volatility (%)'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder = 'normal',
            bgcolor = '#E2E2E2',
            bordercolor = 'black',
            borderwidth = 2),
        width = 800,
        height = 600,
        margin = dict(l=10, r=10, b=10),
        plot_bgcolor='white',  
        paper_bgcolor='white'  
    )
    fig = go.Figure(data = data, layout= layout)
    return fig.show()




ETF_list = ['IVV', 'IJR', 'IEFA', 'IEMG', 'BND', 'BNDX',  'IAU', 'VNQ']
end = dt.datetime.now()
# start = end - dt.timedelta(days=365)
start = end - relativedelta(years=10)

mean_returns, cov_matrix = getData(ETF_list, start, end)

# max_SR(mean_returns, cov_matrix, riskfree=0.035)

EF_graph(mean_returns, cov_matrix, 0.03)


# efficient_frontier(mean_returns,cov_matrix,0.03)


calculated_results(mean_returns, cov_matrix, 0.03)
calculated_table(mean_returns, cov_matrix, 0.03)


