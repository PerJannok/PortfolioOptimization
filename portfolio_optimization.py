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

### initialize  ###
dt_start = dt.datetime(2010, 7, 1)
dt_end = dt.datetime(2015, 1, 1)

#Initialize daily timestamp: closing prices, so timestamp should be hours=16 (STL)
dt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

dataobj = da.DataAccess('Yahoo')
ls_symbols = ['CSC']

initial_wealth = 10000
benchmark = 'SPY'
### ###


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
' Buy or sell according to bollinger bands?
' @param 
' @return dataframe of stocks to trade on tradeDate
'''
def stocksToTradeAccordingToBollingerBands( ls_allsymbols, marketsymbol, startDate, tradeDate ):
    ROLLING_WINDOW = 20;
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

        # Calculate rolling mean and rolling std for the symbol
        close_price = closingPrices[s_sym];
        rolling_mean = pd.rolling_mean(close_price, window=ROLLING_WINDOW);
        rolling_std = pd.rolling_std(close_price, window=ROLLING_WINDOW);

        # Calculate normalized Bollinger values for the symbol
        norm_bvals = (close_price - rolling_mean) / (rolling_std * N_STD_FACTOR);
        
        # Calculating the Bollinger values
        norm_bval_today = norm_bvals.ix[tradeDate];
        daybefore = du.getPrevNYSEday(tradeDate);
        norm_bval_yest = norm_bvals.ix[daybefore];
        
#        # Calculate upper and lower Bollinger bands
#        upper_bollinger = rolling_mean + rolling_std * N_STD_FACTOR;
#        lower_bollinger = rolling_mean - rolling_std * N_STD_FACTOR;
        
        #current price < rolling mean -> down trend
        #current price > rolling mean -> up trend
        
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
        
        
#        if norm_bval_yest >= -2 and norm_bval_today <= -2 and market_norm_bval_today >= 1:
#			row = pd.DataFrame([{'equity': s_sym, 'order': 'Buy'}]);
#            df = df.append(row);
    
    return df;

            
ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
ldf_data_symbols = dataobj.get_data(dt_timestamps, ls_symbols, ls_keys)
d_data_symbols = dict(zip(ls_keys, ldf_data_symbols))

ldf_data_benchmark = dataobj.get_data(dt_timestamps, [benchmark], ['close'])
d_data_benchmark = dict(zip(['close'], ldf_data_benchmark))

for s_key in ls_keys:
	d_data_symbols[s_key] = d_data_symbols[s_key].fillna(method='ffill')
	d_data_symbols[s_key] = d_data_symbols[s_key].fillna(method='bfill')
	d_data_symbols[s_key] = d_data_symbols[s_key].fillna(1.0)

    
df_close_symbols = d_data_symbols['actual_close']

close_data_benchmark = d_data_benchmark['close'].values


# print plot of benchmark closing price over specified date range
#plt.clf()
#plt.plot(dt_timestamps, close_data_benchmark)
#plt.legend(benchmark)
#plt.ylabel('Adjusted Close')
#plt.xlabel('Date')
#plt.savefig('adjustedclose_benchmark.pdf', format='pdf')  


#tradeDate = dt.date.today()    # date object for today
tradeDate =  dt.date(2011, 2, 2);
tradeRange = du.getNYSEdays(dt_start, tradeDate, dt.timedelta(hours=16))


for i in range(1, len(tradeRange)):
    trade_df = stocksToTradeAccordingToBollingerBands( ls_symbols, benchmark, tradeRange[0], tradeRange[i] );
    if not trade_df.empty:
        print str(tradeRange[i]);
        print trade_df;
    

#
# Iterate for investing
#
shares = {}
wealth_history = [None] * len(dt_timestamps)
cash_wealth = initial_wealth
for equity in ls_symbols:
	shares[equity] = 0
j = 0

for i in range(len(dt_timestamps)):
	# while ldt_timestamps[i].date() == orders['date'][j]:
		# current_shares = orders['shares'][j]
		# current_equity = orders['equity'][j]
		# current_order =  orders['order'][j]
		# if current_order == 'Buy':
			# shares[current_equity] += current_shares
			# cash_wealth -= current_shares*close_dataframe[current_equity][i]
		# if current_order == 'Sell':
			# shares[current_equity] -= current_shares
			# cash_wealth += current_shares*close_dataframe[current_equity][i]
		# j += 1
		# if j == len(orders):
			# break
	invested_wealth = 0
	# for equity in symbols:
		# invested_wealth += close_dataframe[equity][i]*shares[equity]
	total_wealth = invested_wealth + cash_wealth
	wealth_history[i] = total_wealth
	# i += 1


#
# Calculate sharpe ratio for our portfolio and the benchmark
#
wealth_history_vector = np.asarray(wealth_history);
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


print "Date Range : " + str(dt_start) + " to " + str(dt_end)
print "Trading days Range : " + str(dt_timestamps[0]) + " to " + str(dt_timestamps[-1])

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
print "Total return (portfolio): " + str(wealth_history[len(wealth_history) - 1] / initial_wealth)
print "Portfolio daily return: " + str(portfolio_avg_daily_ret)
print "Portfolio standard deviation: " + str(portfolio_std_dev)
print "Portfolio Sharpe ratio: " + str(portfolio_sharpe)


#
# Plot Portfolio vs benchmark
#
plt.clf()
plt.setp(plt.xticks()[1], rotation=30)
plt.plot(dt_timestamps, wealth_history_copy)
plt.plot(dt_timestamps, benchmark_cum_return);
plt.legend(['Portfolio', benchmark])
plt.ylabel('Value')
plt.xlabel('Date')
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig('portfolio.pdf', format='pdf')
