# Reproduce Table 4 from Neely, Rapach, Zhou (2011)
#
# Larry Takeuchi
# first version: 12/02/2015
#
# Note: uses simple equity premium (not log equity premium)

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from IPython.display import display, HTML
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from oos_forecast import oos_forecast
from perform_cw_test import perform_cw_test
from perform_asset_allocation import perform_asset_allocation
from oos_pc_forecast import oos_pc_forecast

# -----------------------------------------------------------------------------
# Set parameters

# set date range (Initial: 1950-12 to 1965-12, OOS: 1966-01 to 2011-12 for NRZ)
beg_date_init = '1950-12'
beg_date_oos = '1966-01'
end_date_oos = '2011-12'
#end_date_oos = '2014-12'

# Risk aversion paramter
gamma_MV = 5

# transactions cost in basis points
c_bp = 50

# size of rolling window for vol forecast
window_size = 5 * 12


# -----------------------------------------------------------------------------
# Load data
path_name = '/Users/ltakeuchi/Python/stockpredict/data/'
file_name = 'nrz_fundamentals_monthly.pkl'
#file_name = 'fundamentals_monthly.pkl'
df = pd.read_pickle(path_name + file_name)

# load NBER recession indicator
nber = pd.read_pickle(path_name + 'nber_monthly.pkl')

# add recession dummies to df
df = pd.concat([df, nber], axis=1)

file_name = 'nrz_technical_variables.pkl'
tech = pd.read_pickle(path_name + file_name)
df = pd.concat([df, tech], axis=1)

# change sign so that expected slope coefficents are positive as in NRZ
for i in ['ntis', 'tbl', 'lty', 'infl']:
    df[i] = df[i] * -1

econ_var = ['dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr',
            'tms','dfy','dfr','infl']
tech_var = ['ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12',
            'mom_9', 'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12',
            'vol_3_9', 'vol_3_12']
all_var =  econ_var + tech_var


# get data for specified date range
df_sub = df[beg_date_init:end_date_oos]

# Expanding window historical average forecast for equity premium
df['ha_mean'] = Series(pd.expanding_mean(df_sub['equity_premium']/100,
                min_periods = window_size).shift(1), index = df_sub.index)

# Rolling window historical average forecast for equity premium variance
# note degree of freedom adjusted to match NRZ
df['ha_var'] = Series(pd.rolling_var(df_sub['equity_premium']/100, window_size,
               min_periods = window_size, ddof = 0).shift(1), index = df_sub.index)


# Perform asset allocation using historical average forecasts using c_bp = 0
#  all months
df_sub = df[beg_date_oos:end_date_oos]  
ha_results = perform_asset_allocation(df_sub['equity_premium']/100, df_sub['Rfree'],
	        df_sub['ha_mean'], df_sub['ha_var'], gamma_MV, 0)
#  expansion months
df_exp = df_sub[df_sub['recession']==0]
ha_results_exp = perform_asset_allocation(df_exp['equity_premium']/100, df_exp['Rfree'],
	        df_exp['ha_mean'], df_exp['ha_var'], gamma_MV, 0)
#  expansion months
df_rec = df_sub[df_sub['recession']==1]
ha_results_rec = perform_asset_allocation(df_rec['equity_premium']/100, df_rec['Rfree'],
	        df_rec['ha_mean'], df_rec['ha_var'], gamma_MV, 0)


# create table of results for historical average forecast
ha = {}
for i in ['CER', 'CER_exp', 'CER_rec', 'Sharpe ratio', 'turnover (%)']:
    ha[i] = []
ha['CER'] = ha_results['avg_utility'] * 1200
ha['CER_exp'] = ha_results_exp['avg_utility'] * 1200
ha['CER_rec'] = ha_results_rec['avg_utility'] * 1200
ha['Sharpe ratio'] = ha_results['SR']
ha['turnover (%)'] = ha_results['avg_turnover'] * 100

pd.options.display.float_format = '{:.4f}'.format
table4 = DataFrame(ha, index=['HA'])
table4 = table4[['CER', 'CER_exp', 'CER_rec', 'Sharpe ratio', 'turnover (%)']]
display(table4)


# Perform asset allocation using bivariate regression forecasts

# initialize dictionary of lists
d = {}
for i in ['CER', 'CER_exp', 'CER_rec', 'Sharpe ratio', 'rel turnover',
          'CER w/tc', 'Sharpe w/tc']:
    d[i] = []

# lag the x variables
df[all_var] = df[all_var].shift(1)

#______________________________________________________________________________
# to match NRZ, increment beg_date_init by one month (since we lag the X vars)
beg_date_init = '1951-01' 


for var_list in ['pc_econ', 'pc_tech', 'pc_all']:
    if var_list == 'pc_econ':
        max_k = 3
        prediction = oos_pc_forecast('equity_premium', econ_var, df, max_k, beg_date_init,
            beg_date_oos, end_date_oos)
    elif var_list == 'pc_tech':
        max_k = 1
        prediction = oos_pc_forecast('equity_premium', tech_var, df, max_k, beg_date_init,
            beg_date_oos, end_date_oos)
    elif var_list == 'pc_all':
        max_k = 4
        prediction = oos_pc_forecast('equity_premium', all_var, df, max_k, beg_date_init,
            beg_date_oos, end_date_oos)

    results = perform_asset_allocation(df_sub['equity_premium']/100, df_sub['Rfree'],
                prediction['y_forecast']/100, df_sub['ha_var'], gamma_MV, 0)
    d['CER'].append((results['avg_utility'] - ha_results['avg_utility']) * 1200)
    d['Sharpe ratio'].append(results['SR'])
    d['rel turnover'].append(results['avg_turnover'] / ha_results['avg_turnover'])

    results_exp = perform_asset_allocation(df_exp['equity_premium']/100, df_exp['Rfree'],
        prediction['y_forecast'][df_sub['recession']==0]/100, df_exp['ha_var'], gamma_MV, 0)
    d['CER_exp'].append((results_exp['avg_utility'] - ha_results_exp['avg_utility']) * 1200)

    results_rec = perform_asset_allocation(df_rec['equity_premium']/100, df_rec['Rfree'],
        prediction['y_forecast'][df_sub['recession']==1]/100, df_rec['ha_var'], gamma_MV, 0)
    d['CER_rec'].append((results_rec['avg_utility'] - ha_results_rec['avg_utility']) * 1200)

    # with transactions costs
    results_tc = perform_asset_allocation(df_sub['equity_premium']/100, df_sub['Rfree'],
        prediction['y_forecast']/100, df_sub['ha_var'], gamma_MV, c_bp)
    ha_results_tc = perform_asset_allocation(df_sub['equity_premium']/100, df_sub['Rfree'],
        df_sub['ha_mean'], df_sub['ha_var'], gamma_MV, c_bp)
    d['CER w/tc'].append((results_tc['avg_utility'] - ha_results_tc['avg_utility']) * 1200)
    d['Sharpe w/tc'].append(results_tc['SR'])


# create and display Table 4, panels B and C
table4a = DataFrame(d, index=['pc_econ', 'pc_tech', 'pc_all'])
table4a = table4a[['CER', 'CER_exp', 'CER_rec', 'Sharpe ratio', 'rel turnover',
                   'CER w/tc', 'Sharpe w/tc']]
display(table4a)





