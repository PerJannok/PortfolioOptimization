# TODO

Survivor bias?

Implement trade commision.

Do not reinvest dividends!?
Leave as cash.

RSI indicator?	

# Estimate portfolio returns
cummulative daily return formula:
	daily_cum_ret(t) = daily_cum_ret(t-1) * (1 + daily_ret(t))

daily returns:        
close_px = df['Adj Close']
rets = close_px / close_px.shift(1) - 1    
    or
close_px.pct_change()
    
# portfolio cummulative return, see tutorial 3	
na_portrets = np.sum(na_rets * lf_port_alloc, axis=1)
na_port_total = np.cumprod(na_portrets + 1)
na_component_total = np.cumprod(na_rets + 1, axis=0)

dailyrets = (pricedat[1:,:]/pricedat[0:-1,:]) - 1



### comparison indexes ###

NYSEARCA:SPY                                            - traded fund (pays dividends etc. more realistic comparison)
^GSPC (yahoo), .INX (google), INDEXCBOE:SPX (google)    - actual index
SPLV                                                    - traded fund with only low volatility stocks from S&P500