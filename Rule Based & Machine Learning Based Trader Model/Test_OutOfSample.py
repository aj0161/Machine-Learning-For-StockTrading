import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import util 
import indicators as ind
import RTLearner as rt
import marketsim as m_sim
import benchmark as bm
import BagLearner as bl

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

def get_data(symbols, start_date, end_date):
  dates = pd.date_range(start_date, end_date)

  # Read all the relevant price data (plus SPY) into a DataFrame.
  price = util.get_data(symbols, dates,addSPY=False)
  return price

def rule_testing():
    symbols = ['AAPL']
    lookback = 14

    #Read out of sample data
    bbp, rsi_m, mom, price, smaR, sma_ = ind.get_indicators(symbols, 
                                                            dt.datetime(2010, 01, 01), 
                                                            dt.datetime(2011, 12, 31), 
                                                            lookback)
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
        # (bbp < 0.2 )  and (RSI < 30) and (mom < -0.048) and (sma < 1.0)
        #if (indicators.ix[i,0] < 0.49 and indicators.ix[i,1]< 50 and indicators.ix[i, 2] < -0.059 or indicators.ix[i, 3] < 1.1):
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
        # (bbp > 0.9 )  and (RSI > 50) and (mom > 0.052) and (sma > 1.0)
        elif (indicators.ix[i,0]> 0.5  and indicators.ix[i,1]>70 and indicators.ix[i, 2]> 0.051 or indicators.ix[i, 3] > 1):
        #elif (indicators.ix[i,0]> 0.5  and indicators.ix[i,1]> 51 and indicators.ix[i, 2]> 0.051 or indicators.ix[i, 3] > 1):
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
    
        
    # convert orders list to dataframe and prepare orders.csv file for marketsim
    orders_RB=pd.DataFrame(orders_RB)
    orders_RB=orders_RB.set_index(orders_RB.ix[:,0])
    orders_RB=orders_RB.ix[:,1:5]
    orders_RB.index.name='Date'
    orders_RB.columns=['Symbol','Order','Shares','Enter or Exit']
    of ='orders_RuleBased_TestingData.csv' 
    orders_RB.to_csv(of, sep=',')

    return of
    
def ML_testing():
    symbols = ['AAPL']
    lookback = 14
    bbp, rsi_m, mom, price, smaR, sma_ = ind.get_indicators(symbols, 
                                                            dt.datetime(2008, 01, 01), 
                                                            dt.datetime(2009, 12, 31),
                                                            lookback)
    bbp = bbp['AAPL']
    rsi =  rsi_m['AAPL']
    spy_rsi =  rsi_m['SPY']
    mom =  mom['AAPL']
    price = price['AAPL']
    sma = smaR['AAPL']

    #YBUY=0.03
    #YSELL=-0.03
    YBUY=-0.078
    YSELL=-0.078
    df_temp = pd.concat([sma, bbp, rsi, spy_rsi, mom], keys=['sma', 'bbp', 'rsi', 'spy_rsi', 'mom',], axis=1)
    df_temp.fillna(method='ffill',inplace=True)
    df_temp.fillna(method='bfill',inplace=True)

    #Normalize the indicators so that mean is 0 and STD is 1
    df_temp = (df_temp - np.mean(df_temp, axis=0)) /np.std(df_temp, axis=0)
    df_temp = df_temp.ix[0:-21]

    trainX = df_temp.as_matrix()

    t=0 
    trainY=np.zeros(trainX.shape[0])        
    trainY[t]=1           
    while t<(price.shape[0]-21):
        ret=(price[t+21]/price[t])-1
        if ret> YBUY:
            trainY[t]=1   #BUY
        elif ret < YSELL:
            trainY[t] = -1 # SELL
        else:
            trainY[t] = 0 # do nothing
        t=t+1

    learner = bl.BagLearner(learner = rt.RTLearner, 
                            kwargs = {"leaf_size":5},
                            bags = 15, 
                            boost = False, 
                            verbose = True)

    learner.addEvidence(trainX, trainY)
    #-----------------------------------------------------------------------------------------------------
    #read out sample data
    #-----------------------------------------------------------------------------------------------------
    bbp, rsi_m, mom, price, smaR, sma_ = ind.get_indicators(symbols, 
                                                            dt.datetime(2010, 01, 01), 
                                                            dt.datetime(2011, 12, 31), 
                                                            lookback)
    bbp = bbp['AAPL']
    rsi =  rsi_m['AAPL']
    spy_rsi =  rsi_m['SPY']
    mom =  mom['AAPL']
    price = price['AAPL']
    sma = smaR['AAPL']

    df_test = pd.concat([sma, bbp, rsi, spy_rsi, mom], keys=['sma', 'bbp', 'rsi', 'spy_rsi', 'mom', ], axis=1)
    df_test.fillna(method='ffill',inplace=True)
    df_test.fillna(method='bfill',inplace=True)

    #Normalize the indicators so that mean is 0 and STD is 1
    df_temp = (df_temp - np.mean(df_temp, axis=0)) /np.std(df_temp, axis=0)
    df_temp = df_temp.ix[0:-21]

    testX = df_temp.as_matrix()
    predY = learner.query(testX)
    Y_pred = predY
    Y_pred[np.where(Y_pred<0)]=-1
    Y_pred[np.where(Y_pred>0)]=1
    dates=dates=df_test.index.tolist()
    orders_ML=[]    
    i=0
    while i<(Y_pred.shape[0]):
        if Y_pred[i]==1:
            date=dates[i]
            orders_ML.append([date,'AAPL' ,'BUY', 200, 'Enter'])
            if (i+21<=Y_pred.shape[0]-1): #code for exiting
                date2=dates[i+21]
                orders_ML.append([date2,'AAPL', 'SELL', 200, 'Exit'])
                i=i+21
            else:
                date2=dates[Y_pred.shape[0]-1]
                orders_ML.append([date2,'AAPL', 'SELL', 200, 'Exit'])
                i=i+21
        elif Y_pred[i]==-1:
            date=dates[i]
            orders_ML.append([date,'AAPL' ,'SELL', 200, 'Enter'])
            if (i+21<=Y_pred.shape[0]-1): #code for exiting
                date2=dates[i+21]
                orders_ML.append([date2,'AAPL', 'BUY', 200, 'Exit'])
                i=i+21
            else:
                date2=dates[Y_pred.shape[0]-1]
                orders_ML.append([date2,'AAPL', 'BUY', 200, 'Exit'])
                i=i+21   
        else:
            i=i+1
        
    orders_ML=pd.DataFrame(orders_ML)
    orders_ML=orders_ML.set_index(orders_ML.ix[:,0])
    orders_ML=orders_ML.ix[:,1:4]
    orders_ML.index.name='Date'
    orders_ML.columns=['Symbol',	'Order',	'Shares', 'Enter_Exit']
    of = 'orders_MLBased_TestingData.csv'
    orders_ML.to_csv(of, sep=',')

    return  of

def display_chart(of_ml, of_Rule):
    #--------------------------------------------------------compute portfolio and Save chart-------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------------

    # ML based  
    port_ml = m_sim.compute_portvals(orders_file=of_ml, start_val=100000)
    print port_ml
    port_ml = port_ml / port_ml.ix[0]
    orders=pd.read_csv(of_ml, index_col="Date")


    #benchmark 
    of = "benchmark_test.csv"
    benchmark= bm.compute_portvals(orders_file = of, 
                                   start_val = 100000,
                                   start_date =  dt.datetime(2010, 01, 01),
                                   end_date = dt.datetime(2011, 12, 31))
    print benchmark
    benchmark_N=benchmark/benchmark.ix[0,0]

    #Rule Based
    port_valsRules = m_sim.compute_portvals(orders_file=of_Rule, start_val=100000)
    print port_valsRules
    port_valsRules = port_valsRules / port_valsRules.ix[0]
    orders_RuleBased=pd.read_csv(of_Rule, index_col="Date")

    plt.plot(port_valsRules,color="Blue")
    plt.plot(benchmark_N, color="Black")
    plt.plot(port_ml,color="Green")
    plt.legend(['RuleBased portvolio', 'benchmark', 'MLbased Portvolio'], loc='upper left')

    long_entry = orders.ix[orders.ix[:,1]=='BUY',:]
    long_entry=long_entry.ix[long_entry.ix[:,3]=='Enter', :]
    long_dates_ml=long_entry.index.tolist()

    short_entry = orders.ix[orders.ix[:,1]=='SELL',:]
    short_entry=short_entry.ix[short_entry.ix[:,3]=='Enter', :]
    short_dates_ml=short_entry.index.tolist()

    for LD in long_dates_ml:
        plt.axvline(x=LD, c='g')

    for SD in short_dates_ml:
        plt.axvline(x=SD, c='r')

    plt.savefig('ML_RuleBased_Testing_Data')
    plt.close()

    cr, adr, sddr, sr = get_portvals(port_ml)

    print "ML Based out-of-sample performances"
    print "Sharpe Ratio of ML Based:", sr.values
    print "Volatility (stdev of daily returns)of ML Based:", sddr.values
    print "Average Daily Returnof ML Based:", adr.values
    print "Cumulative Returnof ML Based:", cr.values
    print ""
    print ""

    cr_b, adr_b, sddr_b, sr_b = get_portvals(benchmark_N)
    print "benchmark out-of-sample performances"
    print "Normalized Final Value:", benchmark_N.tail(1).values
    print "Final Value: ",benchmark.tail(1).values 
    print "Sharpe Ratio of benchmark:", sr_b.values
    print "Volatility (stdev of daily returns) of benchmark:", sddr_b.values
    print "Average Daily Return of benchmark:", adr_b.values
    print "Cumulative Return of benchmark:", cr_b.values
    print ""
    print ""

    cr_b, adr_b, sddr_b, sr_b = get_portvals(port_valsRules)
    print "Rule Based out-of-sample performances"
    print "Normalized Final Value:", benchmark_N.tail(1).values
    print "Final Value: ",benchmark.tail(1).values 
    print "Sharpe Ratio of benchmark:", sr_b.values
    print "Volatility (stdev of daily returns) of benchmark:", sddr_b.values
    print "Average Daily Return of benchmark:", adr_b.values
    print "Cumulative Return of benchmark:", cr_b.values
    print ""

if __name__=="__main__":
    """
    #ML testing
    #Run In sample - Add training data to learner [addEvidence] method invoked
    #Run of sample - get testing data and pass the indicator to [Query] method
                   #- generate orderfile and display chart

                    #ML testing
    #Run of sample - get testing data and run it through RuleBased
                   #- generate orderfile and display chart
    """
    orderFile_ml = ML_testing()
    orderFile_rule = rule_testing()
    display_chart(orderFile_ml, orderFile_rule)                    




