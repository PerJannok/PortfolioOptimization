# QSTK Imports
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep
import QSTK.qstkutil.qsdateutil as du

# Third Party Imports

import pandas as pd
import numpy as np
import math
import copy
import datetime as dt
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize


'''
' Calculate Portfolio Statistics 
' @param na_normalized_price: NumPy Array for normalized prices (starts at 1)
' @param lf_allocations: allocation list
' @return list of statistics:
' (Volatility, Average Return, Sharpe, Cumulative Return)
'''
def calcStats(na_normalized_price, lf_allocations):
    #Calculate cumulative daily portfolio value
   	#   Multiply each column by the allocation to the corresponding equity
	#   Sum each row for each day. That is your cumulative daily portfolio value
    na_weighted_price = na_normalized_price * lf_allocations;
    na_portf_value = na_weighted_price.copy().sum(axis=1);

    f_portf_cumrets = np.cumprod(na_portf_value + 1)

    #Calculate daily returns on portfolio
    # Calculate the daily returns of the prices. (Inplace calculation)
    # returnize0 works on ndarray and not dataframes.
    na_portf_rets = na_portf_value.copy()
    tsu.returnize0(na_portf_rets);

    #Calculate volatility (stdev) of daily returns of portfolio
   	# np.std(squareArray) returns the standard deviation of all the elements in the given array
    f_portf_volatility = np.std(na_portf_rets); 

    #Calculate average daily returns of portfolio
    f_portf_avgret = np.mean(na_portf_rets);

	# Sharpe ratio (Always assume you have 252 trading days in an year. And risk free rate = 0) of the total portfolio
    #	Calculate portfolio sharpe ratio (avg portfolio return / portfolio stdev) * sqrt(252)
    f_portf_sharpe = (f_portf_avgret / f_portf_volatility) * np.sqrt(252);

    return [f_portf_volatility, f_portf_avgret, f_portf_sharpe, f_portf_cumrets, na_portf_value];

    
'''
' Buy or sell according to signals?
' @param 
' @return dataframe of stocks to trade on tradeDate
'''
def stocksToTrade( ls_allsymbols, marketsymbol, startDate, tradeDate ):
    ROLLING_WINDOW = 20;
    LONG_ROLLING_WINDOW = 50;
    N_STD_FACTOR = 2;
    
    # Create empty dataframe
    df_columns = ['equity', 'order']
    df = pd.DataFrame(columns = df_columns)

    
    ldt_timestamps = du.getNYSEdays(startDate, tradeDate, dt.timedelta(hours=16));

    ls_allsymbols.append(marketsymbol)
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close'];
    ldf_data_symbols = dataobj.get_data(ldt_timestamps, ls_allsymbols, ls_keys);
    d_data_symbols = dict(zip(ls_keys, ldf_data_symbols));
    ls_allsymbols.remove(marketsymbol)
    
    closingPrices = d_data_symbols['actual_close'];

    # Calculate rolling mean and rolling std for the market
    market_close_price = closingPrices[marketsymbol];
    market_rolling_mean = pd.rolling_mean(market_close_price, window=ROLLING_WINDOW);
    market_rolling_std = pd.rolling_std(market_close_price, window=ROLLING_WINDOW);

    # Calculate normalized Bollinger values for the market
    market_norm_bvals = (market_close_price - market_rolling_mean) / (market_rolling_std * N_STD_FACTOR);
    
    market_norm_bval_today = market_norm_bvals[tradeDate];

    for s_sym in ls_allsymbols:

        # Calculate (simple moving average) rolling mean and rolling std for the symbol
        close_price = closingPrices[s_sym];
        rolling_mean = pd.rolling_mean(close_price, window=ROLLING_WINDOW);
        rolling_std = pd.rolling_std(close_price, window=ROLLING_WINDOW);
        
        # Moving Average Convergence Divergence (MACD)
        # Looks at difference between two moving averages (one short and one long).
        # Uses an EMA (exponential moving average) period. 
        emaSlow = pd.ewma(close_price, span=26);    # long
        emaFast = pd.ewma(close_price, span=12);    # short
        macd = emaFast - emaSlow;
        ema9 = pd.ewma(macd, span=9);        # EMA for short - long
        
        # Calculate long moving average
        long_rolling_mean = pd.rolling_mean(close_price, window=LONG_ROLLING_WINDOW);
        
        # Calculate normalized Bollinger values for the symbol
        norm_bvals = (close_price - rolling_mean) / (rolling_std * N_STD_FACTOR);
        
        # Calculating the Bollinger values
        # > 1 -> above upper Bollinger band
        # < -1 -> below lower Bollinger band
        norm_bval_today = norm_bvals.ix[tradeDate];
        daybefore = du.getPrevNYSEday(tradeDate);
        norm_bval_yest = norm_bvals.ix[daybefore];
        
        sell = False;
        buy = False;
        
        # Simple moving average
        #current price < simple moving average(rolling mean) -> down trend
        #current price > simple moving average(rolling mean) -> up trend
        if close_price[tradeDate] > rolling_mean[tradeDate] and close_price[daybefore] < rolling_mean[daybefore]:
            # up trend
            buy = True;
        
        if close_price[tradeDate] < rolling_mean[tradeDate] and close_price[daybefore] > rolling_mean[daybefore]:
            # down trend
            sell = True;
        
        
        # Long Simple Moving Average & Short Simple Moving average
        if rolling_mean[tradeDate] > long_rolling_mean[tradeDate] and rolling_mean[daybefore] <= long_rolling_mean[daybefore]:
            # up trend
            buy = True;
        
        if rolling_mean[tradeDate] < long_rolling_mean[tradeDate] and rolling_mean[daybefore] >= long_rolling_mean[daybefore]:
            # down trend
            sell = True;
        
        
        # MACD
        # A buying signal is gotten from MACD when the MACD line crosses the 9-day trigger EMA. 
        # when the MACD falls below the signal line, it is a bearish signal, which indicates that it may be time to sell
        if macd[tradeDate] > ema9[tradeDate] and macd[daybefore] <= ema9[daybefore]:
            # buy
            buy = True;
        if macd[tradeDate] < ema9[tradeDate] and macd[daybefore] >= ema9[daybefore]:
            # sell
            sell = True;
        # When the MACD is above zero, the short-term average is above the long-term average, which signals upward momentum. The opposite is true when the MACD is below zero.
        if macd[tradeDate] > 0:
            # upward trend
            buy = True;
        if macd[tradeDate] < 0:
            # downward trend
            sell = True;
        # Buy signal when fast crosses above slow average. Fast moving average goes down under slow -> sell signal.
        if emaSlow[tradeDate] < emaFast[tradeDate] and emaSlow[daybefore] >= emaFast[daybefore]:
            # buy
            buy = True;
        if emaSlow[tradeDate] > emaFast[tradeDate] and emaSlow[daybefore] <= emaFast[daybefore]:
            # sell
            sell = True;
        

        # Bollinger bands
        #current price < lower bollinger band -> potential buy signal
        #current price > upper band -> Can expect a decline soon.
        if norm_bval_yest >= -1 and norm_bval_today < -1:
            #potential buy
            row = pd.DataFrame([{'equity': s_sym, 'order': 'Buy'}]);
            df = df.append(row);
            
        if norm_bval_yest <= 1 and norm_bval_today > 1:
            #potential sell
            row = pd.DataFrame([{'equity': s_sym, 'order': 'Sell'}]);
            df = df.append(row);
        
        
    return df;

'''
' Buy or sell according orders in data frame
' @param 
' @return 
'''
def tradeOrders( orders, cash_wealth, shares, close_prices, indexOfDay ):
    for index, row in orders.iterrows():
        # don't access row directly 
        vals = row.values;
        equity = vals[0];
        order = vals[1];
        
        
        price = close_prices[equity][indexOfDay];
        
        if order == 'Buy':
            order_nr_of_shares = 100;
            
            how_much_to_buy_for = MAX_OF_CASH_TO_TRADE * cash_wealth;
            how_many_to_buy = how_much_to_buy_for / price;
            print how_many_to_buy;
            
            order_size = order_nr_of_shares*price;
            # only buy if enough cash
            if cash_wealth > (order_size + TRADE_COMMISION):
                #print "buy at " + str(price);
                cash_wealth -= TRADE_COMMISION;
                shares[equity] += order_nr_of_shares;
                cash_wealth -= order_nr_of_shares*price;
        
        if order == 'Sell':
            # add some logic about how many to sell?
            #order_nr_of_shares = 100;   
            #shares[equity] -= order_nr_of_shares;
            
            # for now sell all of it
            #print "sell at " + str(price);
            if shares[equity] > 0:
                cash_wealth += shares[equity]*price;
                cash_wealth -= TRADE_COMMISION;
                shares[equity] = 0;
            
    
    return (shares, cash_wealth);
    

    
### initialize  ###
dt_start = dt.datetime(2011, 7, 1);
tradeDate =  dt.date(2012, 10, 4);
#tradeDate = dt.date.today()

dataobj = da.DataAccess('Yahoo');
ls_symbols = ['CSC', 'CSX', 'CHRW'];

initial_wealth = 100000;
TRADE_COMMISION = 100;
MAX_OF_CASH_TO_TRADE = 0.2;
MIN_TRADE_SIZE = 10000;
benchmark = 'SPY';
### ###
    
tradeRange = du.getNYSEdays(dt_start, tradeDate, dt.timedelta(hours=16));
    
ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
ldf_data_symbols = dataobj.get_data(tradeRange, ls_symbols, ls_keys)
d_data_symbols = dict(zip(ls_keys, ldf_data_symbols))

ldf_data_benchmark = dataobj.get_data(tradeRange, [benchmark], ['close'])
d_data_benchmark = dict(zip(['close'], ldf_data_benchmark))

for s_key in ls_keys:
	d_data_symbols[s_key] = d_data_symbols[s_key].fillna(method='ffill')
	d_data_symbols[s_key] = d_data_symbols[s_key].fillna(method='bfill')
	d_data_symbols[s_key] = d_data_symbols[s_key].fillna(1.0)

    
df_close_symbols = d_data_symbols['actual_close']

close_data_benchmark = d_data_benchmark['close'].values



na_price = d_data_symbols['actual_close'].values;
normalized_prices = na_price / na_price[0, :];
plt.clf();
plt.plot(tradeRange, normalized_prices);
plt.legend(ls_symbols);
plt.ylabel('Adjusted Close');
plt.xlabel('Date');
plt.savefig('adjustedclose.pdf', format='pdf');



#
# Initialize investing
#
cash_wealth = initial_wealth;
shares = {}
for equity in ls_symbols:
    shares[equity] = 0;


# back testing from dt_start until tradeDate
wealth_history = [None] * len(tradeRange);
wealth_history[0] = cash_wealth;
for i in range(1, len(tradeRange)):
    #print "date: " + str(tradeRange[i]);
    #print "current shares: " + str(shares[equity]);
    #print "cash: " + str(cash_wealth);
    #do any trade
    trade_df = stocksToTrade( ls_symbols, benchmark, tradeRange[0], tradeRange[i] );
    if not trade_df.empty:
        (shares, cash_wealth) = tradeOrders(trade_df, cash_wealth, shares, df_close_symbols, i);
        #print "shares after trade: " + str(shares[equity]);
        #print "cash after trade: " + str(cash_wealth);
    #calculate wealth
    invested_wealth = 0
    for equity in shares:
        #print "invested in " + equity;
        #print "at current price " + str(df_close_symbols[equity][i]);
        #print "nr of stocks " + str(shares[equity]);
        invested_wealth += df_close_symbols[equity][i]*shares[equity];
    total_wealth = invested_wealth + cash_wealth;
    #print "total wealth: " + str(total_wealth);
    wealth_history[i] = total_wealth;
    
## to check if invest today, set todays date above in initialization
#trade_df = stocksToTrade( ls_symbols, benchmark, tradeRange[0], tradeRange[-1] );
    

#
# Calculate sharpe ratio for our portfolio and the benchmark
#
wealth_history_vector = np.asarray(wealth_history, dtype=float);
wealth_normalized_history = wealth_history_vector / wealth_history_vector[0];
wealth_history_copy = wealth_history_vector.copy();
# Calculate the daily returns of the prices. (Inplace calculation)
# returnize0 works on ndarray and not dataframes.
tsu.returnize0(wealth_history_copy);
portfolio_avg_daily_ret = np.mean(wealth_history_copy)
portfolio_std_dev = np.std(wealth_history_copy)
portfolio_sharpe = np.sqrt(252) * portfolio_avg_daily_ret / portfolio_std_dev


#Calculate benchmark data
benchmark_normalized_price = close_data_benchmark / close_data_benchmark[0, :]
lf_benchmarkStats = calcStats(benchmark_normalized_price, [1]);
benchmark_avg_daily_ret = lf_benchmarkStats[1];
benchmark_sharpe = lf_benchmarkStats[2];
benchmark_std_dev = lf_benchmarkStats[0];
benchmark_cum_return = lf_benchmarkStats[4];


#benchmark
print ""
print "### benchmark ###"
print "Using " + benchmark + " as benchmark"
print "Total return (benchmark): " + str(benchmark_cum_return[-1] * initial_wealth)
print "Benchmark daily return: " + str(benchmark_avg_daily_ret)
print "Benchmark standard deviation: " + str(benchmark_std_dev)
print "Benchmark Sharpe ratio: " + str(benchmark_sharpe)


#portfolio
print ""
print "### portfolio ###"
print "Total return (portfolio): " + str(wealth_history_vector[-1])
print "Portfolio daily return: " + str(portfolio_avg_daily_ret)
print "Portfolio standard deviation: " + str(portfolio_std_dev)
print "Portfolio Sharpe ratio: " + str(portfolio_sharpe)


#
# Plot Portfolio vs benchmark
#
plt.clf()
plt.setp(plt.xticks()[1], rotation=30)
plt.plot(tradeRange, wealth_normalized_history)
plt.plot(tradeRange, benchmark_cum_return);
plt.legend(['Portfolio', benchmark])
plt.ylabel('Value')
plt.xlabel('Date')
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig('portfolio.pdf', format='pdf')
