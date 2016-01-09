import pandas as pd
import numpy as np

def perform_asset_allocation(y, rf, y_hat, var_hat, gamma, tc_bp, min_wt=0, max_wt=1.5):
    """ Perform asset allocation evaluation
    Based on matlab program by David Rapach.
     
    Inputs (for all returns 0.01 is 1%):
	   y 	   = T-vector of actual returns
	   rf      = T-vector of risk-free rates
	   y_hat   = T-vector of forecast returns
	   var_hat = T-vector of forecast return variance
	   gamma   = risk aversion coefficient
	   tc_bp   = transactions cost in basis points

	Outputs:
	   avg_utility
	   SR              = Sharpe ratio
	   cum_return      = cumulative (gross) return
	   avg_turnover    = average turnover

    """

    tc_bp = tc_bp / 10000
    T = len(y)

    weight_risky = 1 / gamma * y_hat / var_hat
    weight_risky[weight_risky < min_wt] = min_wt
    weight_risky[weight_risky > max_wt] = max_wt

    wealth_total = 1 + rf + weight_risky * y
    wealth_risky = weight_risky * (1 + rf + y)

    target_risky = weight_risky.shift(-1) * wealth_total
    turnover = np.absolute(target_risky - wealth_risky) / wealth_total
    tc = tc_bp * turnover

    portfolio_return = wealth_total * (1 - tc_bp * turnover) - 1
    portfolio_return[-1] = rf[-1] + weight_risky[-1] * y[-1]

    avg_utility = portfolio_return.mean() - 0.5 * gamma * np.power(portfolio_return.std(), 2)
    excess_portfolio_return = portfolio_return - rf
    SR = excess_portfolio_return.mean() / excess_portfolio_return.std()
    cum_return = np.cumprod(portfolio_return + 1)[-1]

    results = {'avg_utility':avg_utility, 'SR':SR, 'cum_return': cum_return,
               'avg_turnover': turnover.mean()}
    return results