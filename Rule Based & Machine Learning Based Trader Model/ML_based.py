import numpy as np
import pandas as pd
import datetime as dt
import math
import time
import util
import sys
import matplotlib.pyplot as plt
import csv
import indicators as ind
import marketsim  as m_sim
import RTLearner as rt
import BagLearner as bt
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

symbols = ['AAPL']
lookback = 14
bbp, rsi_m, mom, price, smaR, sma_ = ind.get_indicators(symbols, dt.datetime(2008, 01, 01), dt.datetime(2009, 12, 31), lookback)
bbp = bbp['AAPL']
rsi =  rsi_m['AAPL']
spy_rsi =  rsi_m['SPY']
mom =  mom['AAPL']
price = price['AAPL']
sma = smaR['AAPL']

# Get X DATA
df_temp = pd.concat([sma, bbp, rsi, spy_rsi, mom], keys=['sma', 'bbp', 'rsi', 'spy_rsi', 'mom', ], axis=1)
df_temp.fillna(method='ffill',inplace=True)
df_temp.fillna(method='bfill',inplace=True)

#Normalize the indicators so that mean is 0 and STD is 1
df_temp = (df_temp - np.mean(df_temp, axis=0)) /np.std(df_temp, axis=0)
df_temp = df_temp.ix[0:-21]

#Normalize the indicators so that mean is 0 and STD is 1
df_temp = (df_temp - np.mean(df_temp, axis=0)) /np.std(df_temp, axis=0)

testX = df_temp.as_matrix()  #testing data which is same as training data

bbp, rsi_m, mom, price, smaR, sma_ = ind.get_indicators(symbols, dt.datetime(2008, 01, 01), dt.datetime(2009, 12, 31), lookback)
bbp = bbp['AAPL']
rsi =  rsi_m['AAPL']
spy_rsi =  rsi_m['SPY']
mom =  mom['AAPL']
price = price['AAPL']
sma = smaR['AAPL']


#tweak it for better performance
#YBUY=0.03
#YSELL=-0.03
YBUY=-0.078
YSELL=-0.078
df_temp = pd.concat([sma, bbp, rsi, spy_rsi, mom], keys=['sma', 'bbp', 'rsi', 'spy_rsi', 'mom',], axis=1)
df_temp.fillna(method='ffill',inplace=True)
df_temp.fillna(method='bfill',inplace=True)
df_temp = df_temp.dropna()
df_temp = df_temp.ix[0:-21]

#Normalize the indicators so that mean is 0 and STD is 1
df_temp = (df_temp - np.mean(df_temp, axis=0)) /np.std(df_temp, axis=0)

trainX = df_temp.as_matrix()
train_data = df_temp  #part 5

t=0 
trainY=np.zeros(trainX.shape[0])
# classify based on 21 day change in price.
## trainY that reflects the 21 day change and aligns with the current date.        
trainY[t]=1
c=1       
while t<(price.shape[0]-21):

    ret=(price[t+21]/price[t])-1
    if ret> YBUY:
        trainY[t]=1   #BUY
    elif ret < YSELL:
        trainY[t] = -1 # SELL
    else:
        trainY[t] = 0 # do nothing
        c= c+1
    t=t+1


buy_bbp= np.empty((484,))  #bbp
buy_bbp[:] = np.nan
sell_bbp= np.empty((484,))  
sell_bbp[:] = np.nan

buy_sma = np.empty((484,))  #sma
buy_sma[:] = np.nan
sell_sma = np.empty((484,))
sell_sma[:] = np.nan

t=0 
while t < (trainY.shape[0]):
    if trainY[t]== 1:
        buy_bbp[t]=df_temp.bbp[t]
        buy_sma[t]=df_temp.sma[t]
    elif trainY[t]== -1:
         sell_bbp[t]=df_temp.bbp[t]
         sell_sma[t]=df_temp.sma[t]
    else:
        trainY[t] = 0 # do nothing
        trainY[t]=price.index.values[t]
    t=t+1

buy_bbp = buy_bbp[~np.isnan(buy_bbp)] #remove nan
sell_bbp = sell_bbp[~np.isnan(sell_bbp)] #remove nan
buy_sma = buy_sma[~np.isnan(buy_sma)] #remove nan
sell_sma = sell_sma[~np.isnan(sell_sma)] #remove nan



 #train In-sample using RTLearner
#learner = rt.RTLearner(leaf_size=7, verbose=False)
learner = bl.BagLearner(learner = rt.RTLearner, 
                            kwargs = {"leaf_size":5},
                            bags = 15, 
                            boost = False, 
                            verbose = True)
learner.addEvidence(trainX, trainY)  # train it
predY = learner.query(testX)
Y_pred = predY
Y_pred[np.where(Y_pred<0)]=-1
Y_pred[np.where(Y_pred>0)]=1
dates=dates=df_temp.index.tolist()
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
of = 'ordersMLBased.csv'
orders_ML.to_csv(of, sep=',')

#--------------------------------------------------------compute portfolio and Save chart-------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# ML based  
port_ml = m_sim.compute_portvals(orders_file=of, start_val=100000)
print port_ml
port_ml = port_ml / port_ml.ix[0]
orders=pd.read_csv(of, index_col="Date")

#benchmark 
of = "benchmark.csv"
benchmark= bm.compute_portvals(orders_file = of, start_val = 100000)
print benchmark
benchmark_N=benchmark/benchmark.ix[0,0]

#Rule Based
of = "orders_RuleBased.csv"
port_valsRules = m_sim.compute_portvals(orders_file=of, start_val=100000)
print port_valsRules
port_valsRules = port_valsRules / port_valsRules.ix[0]
orders_RuleBased=pd.read_csv(of, index_col="Date")

plt.plot(port_valsRules,color="Blue")
plt.plot(benchmark_N,color="Black")
plt.plot(port_ml,color="Green")
plt.legend(['RuleBased portvolio', 'benchmark', 'MLbased Portvolio'], loc='upper left')

long_entry = orders.ix[orders.ix[:,1]=='BUY',:]
long_entry=long_entry.ix[long_entry.ix[:,3]=='Enter', :]
long_dates=long_entry.index.tolist()

short_entry = orders.ix[orders.ix[:,1]=='SELL',:]
short_entry=short_entry.ix[short_entry.ix[:,3]=='Enter', :]
short_dates=short_entry.index.tolist()

for LD in long_dates:
    plt.axvline(x=LD, c='g')

for SD in short_dates:
    plt.axvline(x=SD, c='r')

plt.savefig('ML_Based_Both')
plt.close()

cr, adr, sddr, sr = get_portvals(port_ml)

print "ML Based Stats"
print "Sharpe Ratio of ML Based:", sr.values
print "Volatility (stdev of daily returns)of ML Based:", sddr.values
print "Average Daily Returnof ML Based:", adr.values
print "Cumulative Returnof ML Based:", cr.values
print ""
print ""

cr_b, adr_b, sddr_b, sr_b = get_portvals(benchmark_N)
print "benchmark Stats"
print "Normalized Final Value:", benchmark_N.tail(1).values
print "Final Value: ",benchmark.tail(1).values 
print "Sharpe Ratio of benchmark:", sr_b.values
print "Volatility (stdev of daily returns) of benchmark:", sddr_b.values
print "Average Daily Return of benchmark:", adr_b.values
print "Cumulative Return of benchmark:", cr_b.values
print ""
print ""

cr_b, adr_b, sddr_b, sr_b = get_portvals(port_valsRules)
print "Rule Based Stats"
print "Normalized Final Value:", benchmark_N.tail(1).values
print "Final Value: ",benchmark.tail(1).values 
print "Sharpe Ratio of benchmark:", sr_b.values
print "Volatility (stdev of daily returns) of benchmark:", sddr_b.values
print "Average Daily Return of benchmark:", adr_b.values
print "Cumulative Return of benchmark:", cr_b.values
print ""



## -------------------------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------
##part 5.2 response from  a learner-------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------

 
#indicators_two= pd.concat([train_data.bbp,  train_data.sma], 
#                          axis=1, 
#                          join_axes=[train_data.bbp.index])
    
## Create data
#N = 505
#x = indicators_two.bbp
#y = indicators_two.sma
#area = np.pi*3

## Plot
#plt.xlim(-1.5, 1.5)
#plt.ylim(-1.5, 1.5)
#plt.scatter(x, y, s=area,color='black')

#plt.scatter(sell_bbp, sell_bbp, c='red', marker="s",s =20)     #SHORT
#plt.scatter(buy_sma, buy_sma, c='Green', marker="s", s =20)     #LONG
#plt.title('Part 5.2: Data Visualization: ML Based Strategy')
#plt.xlabel('Bollinger Band')
#plt.ylabel('sma')
#plt.show()
#print 



## -------------------------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------
##part 5.3 The training for ML strategy-------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------


#indicators_two= pd.concat([train_data.bbp,  train_data.sma], 
#                          axis=1, 
#                          join_axes=[train_data.bbp.index])

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

#plt.scatter(short_data_x, short_data_y, c='red', marker="s",s =20)     #SHORT
#plt.scatter(long_data_x, long_data_y, c='Green', marker="s", s =20)     #LONG
#plt.title('Part 5.3: Data Visualization: ML Based Strategy')
#plt.xlabel('Bollinger Band')
#plt.ylabel('sma')
#plt.show()
#print 


 







