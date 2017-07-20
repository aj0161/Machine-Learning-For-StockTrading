""" Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data


def assess_portfolio(sd =dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), syms=['GOOG','AAPL','GLD','XOM'], allocs=[0.1,0.2,0.3,0.4], sv=1000000, rfr=0.0, sf=252.0,  gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Fill missing values
    prices = fill_missing_values(prices.copy())

    # Get daily portfolio value
    port_val = compute_daily_portfolio_value(prices, allocs, sv)
    # print "port_val:",port_val

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_portfolio_stats(port_val, allocs, rfr = rfr, sf = sf)

    # End value
    ev = port_val/port_val.ix[0]
    ev = ev.tail(1).iloc[0] * sv

    # Print statistics
    print "Start Date:", sd
    print "End Date:", ed
    print "Symbols:", syms
    print "Allocations:", allocs
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr
    print "ev:", ev

    return cr, adr, sddr, sr, ev

#first fill forward and then fill backward
def fill_missing_values(df):
    df.fillna(method='ffill', inplace="TRUE") #fill forward
    df.fillna(method='Bfill', inplace="TRUE") #fill backward
    return df

def test_code():

    start_date =dt.datetime(2002,01,01)
    end_date = dt.datetime(2014,12,07)
    symbols = ['FAKE1', 'FAKE2', 'JAVA', 'SPY']
    allocations = [0.2, 0.2, 0.3, 0.3]
    start_val = 1000000
    risk_free_rate = 0.001
    sample_freq = 252

    # Assess the portfolio
    assess_portfolio(start_date, end_date, symbols, allocations,start_val,rfr=risk_free_rate, sf=sample_freq, gen_plot=False)

# The input parameters are:
# prices is a data frame or an ndarray of historical prices.
# allocs: A list of allocations to the stocks, must sum to 1.0
# rfr: The risk free return per sample period for the entire date range. We assume that it does not change.
# sf: Sampling frequency per year
def compute_portfolio_stats(port_val, allocs, rfr = 0.0, sf = 252.0):
    daily_rets = (port_val/port_val.shift(1))-1
    cr = (port_val[-1]/port_val[0])-1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = ((sf)**.5)*((daily_rets-rfr).mean()/(daily_rets-rfr).std())
    return cr, adr, sddr, sr

# Compute daily portfolio value given stock prices, allocations and starting value.
# Assumption: length of symbol and allocs are same
def compute_daily_portfolio_value(prices, allocs, start_val):
    normed_value = prices/prices.ix[0]
    allocated_value = normed_value * allocs
    position_value = allocated_value * start_val
    port_val = position_value.sum(axis=1)
    return port_val

if __name__ == "__main__":
    test_code()