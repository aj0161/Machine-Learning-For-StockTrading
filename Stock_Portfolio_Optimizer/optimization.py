"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo

def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    allocs = get_Optimal_Allocs(prices)
    allocs = allocs / np.sum(allocs)  # normalize allocations, if they don't sum to 1.0
    # print "sum of allocation: ", sum(allocs)

    # Get daily portfolio value (already normalized since we use default start_val=1.0)
    port_val = compute_daily_portfolio_value(prices, allocs)

    cr, adr, sddr, sr = compute_portfolio_stats(port_val)

    if gen_plot:
        normed_SPY = prices_SPY / prices_SPY.ix[0,:]
        df_temp = pd.concat([port_val, normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, title=" Daily Portfolio Value and SPY")
        pass

    return allocs, cr, adr, sddr, sr


def get_Optimal_Allocs(prices):

    def optimizer_function(allocs):
        port_val=compute_daily_portfolio_value(prices, allocs)
        sharpe_ratio=compute_portfolio_stats(port_val)[3]
        return -( sharpe_ratio)

    bounds=tuple((0, 1) for x in range(prices.shape[1]))
    constraints= ({'type':'eq','fun':lambda x:np.sum(x)-1})
    allocs=prices.shape[1]*[1./prices.shape[1],]

    res=spo.minimize(optimizer_function, allocs, method='SLSQP',bounds=bounds, constraints=constraints)
    return res.x

def test_code():

    start_date = dt.datetime(2008,06,01)
    end_date = dt.datetime(2009,06,01)
    symbols =   ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


def assess_portfolio(sd =dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1),
                     syms=['GOOG','AAPL','GLD','XOM'], allocs=[0.1,0.2,0.3,0.4],
                     sv=1000000, rfr=0.0, sf=252.0,  gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Fill missing values
    prices = fill_missing_values(prices.copy())

    # Get daily portfolio value
    port_val = compute_daily_portfolio_value(prices, allocs)
    # print "port_val:",port_val

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_portfolio_stats(port_val, rfr = rfr, sf = sf)

    return cr, adr, sddr, sr

def compute_daily_portfolio_value(prices, allocs):
    normed_value = prices/prices.ix[0]
    allocated_value = normed_value * allocs
    port_val = allocated_value.sum(axis=1)
    return port_val

def compute_portfolio_stats(port_val, rfr = 0.0, sf = 252.0):
    daily_rets = (port_val/port_val.shift(1))-1
    cr = (port_val[-1]/port_val[0])-1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = ((sf)**.5)*((daily_rets-rfr).mean()/(daily_rets-rfr).std())
    return cr, adr, sddr, sr

def fill_missing_values(df):
    df.fillna(method='ffill', inplace="TRUE") #fill forward
    df.fillna(method='Bfill', inplace="TRUE") #fill backward
    return df

if __name__ == "__main__":
    test_code()