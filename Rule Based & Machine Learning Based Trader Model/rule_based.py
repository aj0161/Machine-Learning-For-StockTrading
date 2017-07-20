import numpy as np
import pandas as pd
import datetime as dt
import math
import time
import util
import sys
import os
import matplotlib.pyplot as plt
import csv
import indicators as ind
import marketsim as m_sim
import benchmark as bm

def get_portvals(prices):
    prices=pd.DataFrame(prices)
    cr=(prices.ix[len(prices)-1]/prices.ix[0])-1
    dr=prices.copy()
    dr[1:]=(prices.ix[1:]/prices.ix[:-1].values)-1
    dr.ix[0]=0
    adr=dr[1:].sum()/(len(dr)-1)
    sddr=dr[1:].std()
    sr=((adr-0)/sddr)*252**0.5
    return cr, adr, sddr, sr

symbols = ['AAPL']
lookback = 14
bbp, rsi_m, mom, price, smaR, sma_  = ind.get_indicators(symbols, dt.datetime(2008, 01, 01), dt.datetime(2009, 12, 31), lookback)
bbp = pd.Series(bbp['AAPL'], name='bbp')
rsi = pd.Series(rsi_m['AAPL'], name='rsi') 
mom = pd.Series(mom['AAPL'], name='mom')  
price = pd.Series(price['AAPL'], name='AAPL') 
sma = pd.Series(smaR['AAPL'], name='sma') 

#dataframe containing all the indicators and adjusted closed prices
indicators= pd.concat([bbp, rsi, mom, sma,price], axis=1, join_axes=[price.index])

orders_RB=[] # orders list - dates symbol order shares (shares=200)
dates=indicators.index.tolist() # list that contains the trading dates


i=0
while i<len(indicators):
    # (bbp < 0.49 )  and (RSI < 30) and (mom < -0.059) and (sma < 1.1)
    if (indicators.ix[i,0] < 0.49 and indicators.ix[i,1]< 30 and indicators.ix[i, 2] < -0.059 or indicators.ix[i, 3] < 1.1):
        date=dates[i]
        orders_RB.append([date,'AAPL' ,'BUY', 200, 'Enter'])
        if (i+21<=len(indicators)-1): #code for exiting
            date2=dates[i+21]
            orders_RB.append([date2,'AAPL', 'SELL', 200, 'Exit'])
            i=i+21
        else:
            date2=dates[(len(indicators)-1)]
            orders_RB.append([date2,'AAPL', 'SELL', 200, 'Exit'])
            i=i+21
    # (bbp > 0.5 )  and (RSI > 70) and (mom > 0.051) and (sma > 1.0)
    elif (indicators.ix[i,0]> 0.5  and indicators.ix[i,1]>70 and indicators.ix[i, 2]> 0.051 or indicators.ix[i, 3] > 1):
        date=dates[i]
        orders_RB.append([date,'AAPL' ,'SELL', 200, 'Enter'])
        if (i+21<=len(indicators)-1): # code for exiting
            date2=dates[i+21]
            orders_RB.append([date2,'AAPL', 'BUY', 200, 'Exit'])
            i=i+21
        else:
            date2=dates[(len(indicators)-1)]
            orders_RB.append([date2,'AAPL', 'BUY', 200, 'Exit'])
            i=i+21
    else:
        i=i+1
    
orders_RB=pd.DataFrame(orders_RB)
orders_RB=orders_RB.set_index(orders_RB.ix[:,0])
orders_RB=orders_RB.ix[:,1:5]
orders_RB.index.name='Date'
orders_RB.columns=['Symbol','Order','Shares','Enter or Exit']
of ='orders_RuleBased.csv' 
orders_RB.to_csv(of, sep=',')


#--------------------------------------------------------compute portfolio and Save chart-------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

#compute portfolio  
portvolio = m_sim.compute_portvals(orders_file = of, start_val = 100000)
print portvolio
#normalized to 1.0 
portvolio = portvolio/portvolio.ix[0,0]
orders=pd.read_csv(of, index_col="Date")


#benchmark 
of = "benchmark.csv"
benchmark= bm.compute_portvals(orders_file = of, start_val = 100000)
print benchmark
#normalized to 1.0 
benchmark_N=benchmark/benchmark.ix[0,0]

long_entry=orders.ix[orders.ix[:,1]=='BUY',:]
long_entry=long_entry.ix[long_entry.ix[:,3]=='Enter', :]
long_dates=long_entry.index.tolist()

short_entry=orders.ix[orders.ix[:,1]=='SELL',:]
short_entry=short_entry.ix[short_entry.ix[:,3]=='Enter', :]
short_dates=short_entry.index.tolist()

plt.plot(portvolio, c='b',color="Blue")
plt.plot(benchmark_N, c='k',color="Black")
plt.legend(['Rule-based Portvolio', 'benchmark'], loc='best')

for LD in long_dates:
    plt.axvline(x=LD, c='g')

for SD in short_dates:
    plt.axvline(x=SD, c='r')

plt.savefig('Rule_Based_Chart')
plt.close()

cr, adr, sddr, sr = get_portvals(portvolio)

print "Sharpe Ratio of Rule Based:", sr.values
print "Volatility (stdev of daily returns)of Rule Based:", sddr.values
print "Average Daily Returnof Rule Based:", adr.values
print "Cumulative Returnof Rule Based:", cr.values
print""
print ""

cr_b, adr_b, sddr_b, sr_b = get_portvals(benchmark_N)

print "Normalized Final Value:", benchmark_N.tail(1).values
print "Final Value: ",benchmark.tail(1).values 
print "Sharpe Ratio of benchmark:", sr_b.values
print "Volatility (stdev of daily returns) of benchmark:", sddr_b.values
print "Average Daily Return of benchmark:", adr_b.values
print "Cumulative Return of benchmark:", cr_b.values
print

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
#part 5 -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------


#indicators_two= pd.concat([indicators.bbp,  indicators.sma], 
#                          axis=1, 
#                          join_axes=[indicators.bbp.index])
#indicators_two.fillna(method='ffill',inplace=True)
#indicators_two.fillna(method='bfill',inplace=True)

##Normalize the indicators 
#indicators_two = (indicators_two - np.mean(indicators_two, axis=0)) /np.std(indicators_two, axis=0)

## Create data
#N = 505
#x = indicators_two.bbp
#y = indicators_two.sma
#area = np.pi*3

#short_data_x= np.empty((505,))
#short_data_x[:] = np.nan
#short_data_y= np.empty((505,))
#short_data_y[:] = np.nan
#for SD in short_dates:
#    result = indicators_two.loc[SD] 
#    short_data_x = np.append(short_data_x, [result.bbp], axis=0)
#    short_data_y = np.append(short_data_y, [result.sma], axis=0)
#short_data_x = short_data_x[~np.isnan(short_data_x)] #remove nan
#short_data_y = short_data_y[~np.isnan(short_data_y)] #remove nan


#long_data_x = np.empty((505,))
#long_data_x[:] = np.nan
#long_data_y = np.empty((505,))
#long_data_y[:] = np.nan
#for SD in long_dates:
#    result = indicators_two.loc[SD] 
#    long_data_x = np.append(long_data_x, [result.bbp], axis=0)
#    long_data_y = np.append(long_data_y, [result.sma], axis=0)
#long_data_x = long_data_x[~np.isnan(long_data_x)] #remove nan
#long_data_y = long_data_y[~np.isnan(long_data_y)] #remove nan

## Plot
#plt.xlim(-1.5, 1.5)
#plt.ylim(-1.5, 1.5)
#plt.scatter(x, y, s=area,color='black')

#plt.scatter(short_data_x, short_data_y, c='red', marker="s",s =25)     #SHORT
#plt.scatter(long_data_x, long_data_y, c='Green', marker="s", s =25)     #LONG
#plt.title('Data Visualization: Rule Based Strategy')
#plt.xlabel('Bollinger Band')
#plt.ylabel('sma')
#plt.show()
#print 