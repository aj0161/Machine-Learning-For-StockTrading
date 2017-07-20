import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import time
import util
import sys

def get_data(symbols= ['AAPL'], start_date= dt.datetime(2006,1,1), end_date = dt.datetime(2009,12,31)):
  dates = pd.date_range(start_date, end_date)
  price = util.get_data(symbols, dates)
  price=price.dropna()
  return price

def get_indicators(symbols, start_date, end_date, lookback, save_image = False):
    # Read all the relevant price data (plus SPY) into a DataFrame.
    price = get_data(symbols, start_date, end_date)

    # Add SPY to the symbol list for convenience.
    symbols.append('SPY')

    ### Calculate SMA-14 over the entire period in a single step.
    #sma = pd.rolling_mean(price,window=lookback, min_periods=lookback)
    sma = price.rolling(window=lookback, center = False).mean()

    ### Calculate Bollinger Bands (14 day) over the entire period.
    #rolling_std = pd.rolling_std(price,window=lookback, min_periods=lookback)
    rolling_std = price.rolling(window=lookback, min_periods=lookback).std()
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)

    #bbp = (price - bottom_band) / (top_band - bottom_band)
    bbp =(price -sma )/(2*rolling_std)
    ### Now we can turn the SMA into an SMA ratio, which is more useful.
    smaR = price / sma

    ### Calculate Momentum (14 day) for the entire date range for all symbols.
    mom = (price / price.shift(lookback - 1)) - 1

    ### Calculate Relative Strength Index (14 day) for the entire date range for all symbols.
    rs = price.copy()
    rsi = price.copy()

    # Calculate daily_rets for the entire period (and all symbols).
    daily_rets = price.copy()
    daily_rets.values[1:, :] = price.values[1:, :] - price.values[:-1, :]
    daily_rets.values[0, :] = np.nan

    # Split daily_rets into a same-indexed DataFrame of only up days and only down days,
    # and accumulate the total-up-days-return and total-down-days-return for every day.
    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()

    # Apply the sliding lookback window to produce for each day, the cumulative return
    # of all up days within the window, and separately for all down days within the window.
    up_gain = price.copy()
    up_gain.ix[:, :] = 0
    up_gain.values[lookback:, :] = up_rets.values[lookback:, :] - up_rets.values[:-lookback, :]

    down_loss = price.copy()
    down_loss.ix[:, :] = 0
    down_loss.values[lookback:, :] = down_rets.values[lookback:, :] - down_rets.values[:-lookback, :]

    # Now we can calculate the RS and RSI all at once.
    rs = (up_gain / lookback) / (down_loss / lookback)
    rsi = 100 - (100 / (1 + rs))
    rsi.ix[:lookback, :] = np.nan

    # An infinite value here indicates the down_loss for a period was zero (no down days), in which
    # case the RSI should be 100 (its maximum value).
    rsi[rsi == np.inf] = 100


    if save_image == True:
        # SPY col
        data = price.drop('SPY', 1)

        plt.clf()
        plt.cla()
        plt.close()
        #plot SMA
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(data.index,data)
        ax.plot(data.index,sma['AAPL'])
        ax.legend(['price', 'sma'],loc='best')
        ax.set_title('Indicator SMA')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1 = fig.add_subplot(212)
        ax1.plot(data.index,smaR['AAPL'],color="Red")
        ax1.legend(['(data / sma)'], loc='best')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.savefig('SMA')
        plt.close()


      # plot bbp
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(data.index,data)
        ax.plot(data.index,top_band['AAPL'])
        ax.plot(data.index,bottom_band['AAPL'])
        ax.legend(['Upper','Price', 'Lower'], loc='best')
        ax.set_title('Indicator: Bollinger Bands')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1 = fig.add_subplot(212)
        ax1.plot(data.index,bbp['AAPL'],color="Green")
        ax1.axhline(y=1, color='r')
        ax1.axhline(y=-1, color='r')
        ax1.legend(['Bollinger Bands'], loc='best')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.savefig('Bollinger_Bands')
        plt.close()

        #plot RSI
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(data.index,data)
        ax.legend(['price'],loc='best')
        ax.set_title('Indicator RSI')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1 = fig.add_subplot(212)
        ax1.plot(data.index,rsi['AAPL'],color="Red")
        ax1.legend(['RSI'],loc='best')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.savefig('RSI')
        plt.close()

        # plot mom, then price
        plt.subplot(2, 1, 1)
        plt.plot(data)
        plt.legend(['Price'], loc='upper left')
        ax.set_title('Indicator Momentum')
        plt.subplot(2, 1, 2)
        plt.plot(mom['AAPL'],color="Red")
        plt.legend(['Momentum'], loc='upper left')
        plt.savefig('momentum')
        plt.close()

    return bbp, rsi, mom, price, smaR,sma

### Main function.  
#Not called if imported elsewhere as a module. 
#Creates charts for indicators.
if __name__ == "__main__":
    start_date = dt.datetime(2008,01,01) 
    end_date = dt.datetime(2009,12,31)   
    symbols = ['AAPL']
    lookback = 14

    bbp, rsi, mom, price, smaR,sma = get_indicators(symbols, 
                                                    dt.datetime(2008, 01, 01), 
                                                    dt.datetime(2009, 12, 31), 
                                                    lookback,
                                                    save_image = True)

    print



