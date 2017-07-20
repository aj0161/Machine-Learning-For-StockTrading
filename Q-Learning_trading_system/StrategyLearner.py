

import datetime as dt
import QLearner as ql
import pandas as pd
from util import get_data
import numpy as np
import time

# Actions constant
BUY = 1
SELL = 2
NOTHING = 0

class StrategyLearner(object):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def author(self):
        return 'ajoshi319'

    def discretize_data(self,df):
        return pd.qcut(df.rank(method='first'), 5, labels=False, retbins=False)

    def get_state(self, holding, ind1, ind2, ind3, i):
        return 1000 * holding + 100 * ind1[i] + 10 * ind2[i] + ind3[i]

    def get_indicators(self,symbols, start_date, end_date, lookback=10):

        # Read all the relevant price data (plus SPY) into a DataFrame.
        price = get_data([symbols], pd.date_range(start_date, end_date),addSPY=False).dropna()

        sma = pd.rolling_mean(price, window=lookback)

        rolling_std = pd.rolling_std(price, window=lookback)

        top_band = sma + (2 * rolling_std)
        bottom_band = sma - (2 * rolling_std)

        bbp =(price -sma )/(2*rolling_std)
        smaR = price / sma

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

        self.fill_missing_values(smaR)
        self.fill_missing_values(bbp)
        self.fill_missing_values(rsi)

        return price, smaR, bbp, rsi 

    #first fill forward and then fill backward
    def fill_missing_values(self,df):
        df.fillna(method='ffill', inplace="TRUE") #fill forward
        df.fillna(method='Bfill', inplace="TRUE") #fill backward
        return df

    # train trading data in reinforcement learning (Q-learner)
    def addEvidence(self, symbol="AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):

        start = time.time()
        self.learner = ql.QLearner(num_states=3000, num_actions=3)
        prices ,smaR, bbp, rsi = self.get_indicators(symbol,sd, ed)

        prev_cum_return = 0
        sym = 0 
        smaR = self.discretize_data(smaR)
        bbp = self.discretize_data(bbp) 
        rsi = self.discretize_data(rsi)

        
        for i in xrange(25):
            holding = NOTHING
            state = self.get_state(holding, rsi, bbp, smaR, 0)
            action = self.learner.querysetstate(state)
            cash = sv
            for day in xrange(len(prices)):
                reward = NOTHING

                #Implement the action the learner returned (BUY, SELL, NOTHING), and update portfolio value
                holding, reward, cash = self.update_states(prices, holding, day, sym, reward, cash, action) 
             
                #Compute the current state (including holding)
                state_prime = self.get_state(holding, rsi, bbp, smaR, day)

                #Query the learner with the current state and reward to get an action
                action = self.learner.query(s_prime=state_prime, r=reward)
            
            #loop end
            if holding == 1: stock = -200
            elif holding == 2: stock = 200
            else: stock = 0

            current_cum_return = ( (prices.ix[-1,0]*stock + cash) / sv) - 1
            print "current_cum_return is :", current_cum_return

            #30 sec time limit
            stop_time = time.time() - start
            if  stop_time > 24: break

            #Repeat the above loop multiple times until cumulative return stops improving
            if i > 50 and prev_cum_return >= current_cum_return: break
            prev_cum_return = current_cum_return
            
    def testPolicy(self, symbol,sd,ed,sv):
        prices,smaR, bbp, rsi = self.get_indicators(symbol,sd, ed)

        smaR = self.discretize_data(smaR)
        bbp = self.discretize_data(bbp) 
        rsi = self.discretize_data(rsi)

        df_trades = pd.DataFrame(0, index=prices.index,columns=[symbol])
        holding = NOTHING

        for j in range(len(prices)):
            state_prime = self.get_state(holding, rsi, bbp, smaR, j)
            action = self.learner.querysetstate(s=state_prime)

            if action == SELL:
                if holding == 1:  df_trades.ix[j] = 400
                elif holding == 0:df_trades.ix[j] = 200
                holding = 2

            elif action == BUY :
                if holding == 2: df_trades.ix[j] = -400
                elif holding==0: df_trades.ix[j] = -200
                holding = 1

        return df_trades

    #POSITIONS: buy 400, sell 400, buy 200, sell 200, do nothing
    def update_states(self, prices, holding, day, sym, reward, cash, action):
         
         if action == SELL :
             if holding == 1:
                 cash -= 400 * prices.ix[day-1,sym]
                 reward = 400 * (prices.ix[day,sym] - prices.ix[day-1,sym])
                 
             elif holding == 0:
                 cash -= 200 * prices.ix[day - 1,sym]
                 reward = 200 * (prices.ix[day,sym] - prices.ix[day-1,sym])
                  
             else:
                 reward = 200 * (prices.ix[day, sym] - prices.ix[day-1])
             holding = 2

         elif action == BUY :
             if holding == 2:
                 cash += 400 * prices.ix[day - 1,sym]
                 reward = 400 *  (prices.ix[day - 1,sym] - prices.ix[day,sym])
                 
             elif holding== 0:
                 cash += 200 * prices.ix[day - 1,sym]
                 reward = 200 *  (prices.ix[day - 1,sym] - prices.ix[day,sym])
                   
             else:
                 reward = 200 *  (prices.ix[day - 1,sym] - prices.ix[day,sym])
             holding = 1

         elif action == NOTHING :
             if holding == 2:reward = 200 *  (prices.ix[day,sym] - prices.ix[day - 1,sym])
             elif holding == 1:reward = 200 *  (prices.ix[day - 1,sym] - prices.ix[day,sym])
             else:pass
         return holding, reward, cash

if __name__ == "__main__":
    print ""