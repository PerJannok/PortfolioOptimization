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
dt_start = dt.datetime(2013, 1, 1)
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
    #row-wise multiplication by weights
    na_weighted_price = na_normalized_price * lf_allocations;
    #row-wise sum
    na_portf_value = na_weighted_price.copy().sum(axis=1);

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

	# Cumulative return of the total portfolio
    #	Calculate cumulative daily return
    #	...using recursive function
    def cumret(t, lf_returns):
        #base-case
        if t==0:
            return (1 + lf_returns[0]);
        #continuation
        return (cumret(t-1, lf_returns) * (1 + lf_returns[t]));
    f_portf_cumrets = cumret(na_portf_rets.size - 1, na_portf_rets);

    return [f_portf_volatility, f_portf_avgret, f_portf_sharpe, f_portf_cumrets, na_portf_value];


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
