# python 

check syntax
    python -m py_compile <pythonfile>

# PortfolioOptimization


### To update csv files with latest data ###
open cmd window as administrator
set PATH=%PATH%;"C:\Program Files (x86)\python2.7.3"
verify symbol that needs to be updated is included in C:\Program Files (x86)\python2.7.3\Lib\site-packages\QSTK\qstktools\symbol.txt
(I have included S&P 500 index (^GSPC) and all S&P 100 stocks)
cd "C:\Program Files (x86)\python2.7.3\Lib\site-packages\QSTK\qstktools"
python.exe YahooDataPull.py

	

# Estimate portfolio returns
cummulative daily return formula:
	daily_cum_ret(t) = daily_cum_ret(t-1) * (1 + daily_ret(t))

# portfolio cummulative return, see tutorial 3	
na_portrets = np.sum(na_rets * lf_port_alloc, axis=1)
na_port_total = np.cumprod(na_portrets + 1)
na_component_total = np.cumprod(na_rets + 1, axis=0)

dailyrets = (pricedat[1:,:]/pricedat[0:-1,:]) - 1



### comparison indexes ###

NYSEARCA:SPY                                            - traded fund (pays dividends etc. more realistic comparison)
^GSPC (yahoo), .INX (google), INDEXCBOE:SPX (google)    - actual index
SPLV                                                    - traded fund with only low volatility stocks from S&P500