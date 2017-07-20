"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import math

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 100000,  
                     start_date = dt.datetime(2008, 01, 01), end_date = dt.datetime(2009, 12, 31)):
    df_orders_file = pd.read_csv(orders_file,index_col=0, parse_dates=True)
    df_orders_file = df_orders_file.sort()  #sort earliest date first
    my_protVal = get_portVal(df_orders_file, start_date = start_date, end_date=end_date)   
    return my_protVal

def author(self):
    return 'ajoshi319'

#first fill forward and then fill backward
def fill_missing_values(df):
    df.fillna(method='ffill', inplace="TRUE") #fill forward
    df.fillna(method='Bfill', inplace="TRUE") #fill backward
    return df

def compute_portfolio_stats(daily_port_value):
    daily_returns = daily_port_value.copy()
    daily_returns[1:] = (daily_returns[1:] / daily_returns[:-1].values) - 1
    daily_returns.ix[0, :] = 0
    daily_returns = daily_returns.ix[1:]
    adr = daily_returns.mean().iloc[0]
    sddr = daily_returns.std().iloc[0]
    sr = (adr/sddr) * math.sqrt(252)
    cr = (daily_port_value.iloc[-1, 0] / daily_port_value.iloc[0, 0]) - 1
    return cr, adr, sddr, sr

def get_portVal(df_orders, start_val = 100000 , start_date =  dt.datetime(2008, 01, 01), end_date = dt.datetime(2009, 12, 31)):

    symbols_list = df_orders.ix[:,0:1]   #  get all symbols 
    unique_symbols_list = np.unique(symbols_list.values)

    # Set Sell Share num - negative
    df_orders["Shares"] = df_orders.apply(lambda row: row["Shares"] * (-1 if row["Order"] == "SELL" else 1), axis=1)
    df_orders= df_orders.pivot_table(values='Shares',index=df_orders.index.get_values(), columns='Symbol',aggfunc=sum,dropna=True)

    #start and end dates
    dates = df_orders.index.values
    
    #start_date =  dt.datetime(2008, 01, 01) 
    #start_date = pd.Timestamp(start_date).to_pydatetime()

    #end_date = dt.datetime(2009, 12, 31)
    #end_date = pd.Timestamp(end_date).to_pydatetime()

    dfprices = get_data(unique_symbols_list.tolist(), pd.date_range(start_date, end_date),addSPY=False)

    #just get the APPL price from the data range

    dfprices = fill_missing_values(dfprices.copy())   # Fill missing values
    dfprices=dfprices.dropna()   #drop nan

    dfprices['CASH']=1 

    # copy of dfprices and named it df_trade [all Zeros]  - this will represents changes in the number of shares on particular days on each of those assets
    df_trades= pd.DataFrame(data=0, columns=dfprices.columns,index=dfprices.index)
    df_trades= (df_trades + df_orders).fillna(0)

    df_trades['CASH']= -1 * (df_trades[unique_symbols_list] * dfprices[unique_symbols_list]).sum(axis=1)
    
    df_holdings =  df_trades.copy()
    df_holdings["Port_val"] = np.nan

    #only invest on first day
    df_trades_benchmark= pd.DataFrame(data=0, columns=dfprices.columns,index=dfprices.index)
    df_trades_benchmark= (df_trades_benchmark + df_orders).fillna(0)
    df_trades_benchmark['CASH']= -1 * (df_trades['AAPL'][0] * dfprices['AAPL'][0]) #invest first day

    #Benchmark: The performance of a portfolio starting with $100,000 cash, 
    #investing in 200 shares of AAPL and holding that position
    df_holdings_benchmark =  df_trades_benchmark.copy()
    df_holdings_benchmark["Port_val"] = np.nan

    df_holdings_benchmark_shareVal =  df_trades_benchmark.copy()
    df_holdings_benchmark_shareVal["values_shares"] = np.nan

    for i in range(0,df_holdings.shape[0]): #iterate every row 
       if i==0:
           df_holdings['CASH'][i] = df_holdings['CASH'][i] + start_val
           #  investing in 200 shares of AAPL and holding that position
           df_holdings_benchmark['CASH'][i] = df_holdings['CASH'][i] + start_val
       else:
           df_holdings['CASH'][i] = df_holdings['CASH'][i] + df_holdings['CASH'][i-1]
           for symbol in unique_symbols_list: 
               df_holdings[symbol][i] = df_holdings[symbol][i] + df_holdings[symbol][i-1]

       longs = 0
       shorts = 0
       total_shares_val=0

       for symbol in unique_symbols_list: 
           stock_price = dfprices[symbol][df_holdings.index[i]]
           num_of_share =  df_holdings[symbol].values[i]
           shares_for_each_symbol = stock_price * num_of_share
           total_shares_val += shares_for_each_symbol
           if num_of_share > 0:
                longs += shares_for_each_symbol
           else:
                shorts += shares_for_each_symbol

       leverage = (longs + abs(shorts)) / (longs - abs(shorts) + df_holdings['CASH'][i])
       if leverage >= 1.5:
            total_shares_val = 0
            for symbol in unique_symbols_list: 
               df_holdings['CASH'][i] = df_holdings['CASH'][i-1] #cash revert
               df_holdings[symbol].values[i] = df_holdings[symbol].values[i-1] # symbol revert

               stock_price = dfprices[symbol][df_holdings.index[i]]
               num_of_share =  df_holdings[symbol].values[i]
               shares_for_each_symbol = stock_price * num_of_share
               total_shares_val += shares_for_each_symbol

       values_shares=total_shares_val
       df_holdings["Port_val"][i] =  df_holdings['CASH'][i]  + values_shares
       df_holdings_benchmark["Port_val"][i] =  df_holdings_benchmark['CASH'][i]  + values_shares
       df_holdings_benchmark_shareVal["values_shares"][i] = values_shares
    portvals = pd.DataFrame(df_holdings["Port_val"], index = df_holdings.index)
    benchmark_portVal = pd.DataFrame(df_holdings_benchmark["Port_val"], index = df_holdings_benchmark.index)
    return portvals  

def test_code(of):
    sv = 100000     

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
        print "portvals", portvals
        print "end"
    else:
        "warning, code did not return a DataFrame"

if __name__ == "__main__":
    of = "benchmark.csv"
    if os.path.isfile(of) == False:
        print "benchmark.csv is not found!. The file is manually generated."

    test_code(of)