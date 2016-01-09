# Reproduce Table 3 from Neely, Rapach, Zhou (2011)
#
# Larry Takeuchi
# first version: 11/22/2015
#

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from IPython.display import display, HTML
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from oos_forecast import oos_forecast
from oos_pc_forecast import oos_pc_forecast
from perform_cw_test import perform_cw_test


# set date range (Initial: 1950-12 to 1965-12, OOS: 1966-01 to 2011-12 for NRZ)
beg_date_init = '1950-12'
beg_date_oos = '1966-01'
end_date_oos = '2011-12'
#end_date_oos = '2014-12'

# load data
path_name = '/Users/ltakeuchi/Python/stockpredict/data/'
file_name = 'nrz_fundamentals_monthly.pkl'
df = pd.read_pickle(path_name + file_name)

file_name = 'nrz_technical_variables.pkl'
tech = pd.read_pickle(path_name + file_name)

# load NBER recession indicator
nber = pd.read_pickle(path_name + 'nber_monthly.pkl')

# add recession dummies to df
df = pd.concat([df, nber], axis=1)
df = pd.concat([df, tech], axis=1)

# change sign so that expected slope coefficents are positive as in NRZ
for i in ['ntis', 'tbl', 'lty', 'infl']:
    df[i] = df[i] * -1

# var_list = ['dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr',
#             'tms','dfy','dfr','infl']
econ_var = ['dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr',
            'tms','dfy','dfr','infl']
tech_var = ['ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12',
            'mom_9', 'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12',
            'vol_3_9', 'vol_3_12']
var_list = econ_var + tech_var


# get data for specified date range
df_sub = df[beg_date_init:end_date_oos]

# historical average (ha) forecast
init_obs = len(pd.date_range(beg_date_init, beg_date_oos, freq='M')) # Should be 181 obs
ha_forecast = pd.expanding_mean(df_sub['log_equity_premium'], 
	          min_periods=init_obs)
ha_forecast = ha_forecast.shift(1)
ha_err = df_sub['log_equity_premium'][beg_date_oos:end_date_oos] - \
    ha_forecast[beg_date_oos:end_date_oos]
ha_msfe = np.mean(np.power(ha_err, 2))
ha_msfe_exp = np.mean(np.power(ha_err[df_sub['recession'] == 0], 2))
ha_msfe_rec = np.mean(np.power(ha_err[df_sub['recession'] == 1], 2))

# initialize dictionary of lists
d = {}
for i in ['msfe', 'msfe_exp', 'msfe_rec', 'msfe_adj', 'p_value', 'r2', 'r2_exp',
          'r2_rec', 'sq bias', 'rem term']:
    d[i] = []

# lag the x variables
df[var_list] = df[var_list].shift(1)

# to match NRZ, increment beg_date_init by one month
beg_date_init = '1951-01'

for x_var in var_list + ['pc_econ', 'pc_tech', 'pc_all']:
    if x_var == 'pc_econ':
        max_k = 3
        results = oos_pc_forecast('log_equity_premium', econ_var, df, max_k, beg_date_init,
            beg_date_oos, end_date_oos)
    elif x_var == 'pc_tech':
        max_k = 1
        results = oos_pc_forecast('log_equity_premium', tech_var, df, max_k, beg_date_init,
            beg_date_oos, end_date_oos)
    elif x_var == 'pc_all':
        max_k = 4
        results = oos_pc_forecast('log_equity_premium', var_list, df, max_k, beg_date_init,
            beg_date_oos, end_date_oos)
    else:
        results = oos_forecast('log_equity_premium', x_var, df, beg_date_init,
            beg_date_oos, end_date_oos)

    d['msfe'].append(np.mean(np.power(results['res'], 2)))
    d['msfe_exp'].append(np.mean(np.power(results['res'][df['recession'] == 0], 2)))
    d['msfe_rec'].append(np.mean(np.power(results['res'][df['recession'] == 1], 2)))
    d['sq bias'].append(np.power(np.mean(results['res']), 2))
    d['rem term'].append(d['msfe'][-1] - d['sq bias'][-1])
    d['r2'].append((1 - d['msfe'][-1] / ha_msfe) * 100)
    d['r2_exp'].append((1 - d['msfe_exp'][-1] / ha_msfe_exp) * 100)
    d['r2_rec'].append((1 - d['msfe_rec'][-1] / ha_msfe_rec) * 100)

    y = df['log_equity_premium'][beg_date_oos:end_date_oos]
    cw_stat = perform_cw_test(y, ha_forecast[beg_date_oos:end_date_oos], results['y_forecast'])
    d['msfe_adj'].append(cw_stat['MSPE_adj'])
    d['p_value'].append(cw_stat['p_value'])


# 181 initial period
# 552 obs OOS
# 733 total for '1950-12' to '2011-12'


# create and display Table 2, panel A
pd.options.display.float_format = '{:.2f}'.format
table3a = DataFrame(d, index=var_list + ['pc_econ', 'pc_tech', 'pc_all'])
table3a = table3a[['msfe', 'msfe_exp', 'msfe_rec', 'msfe_adj', 'p_value', 'r2', 'r2_exp',
                   'r2_rec', 'sq bias', 'rem term']]
display(table3a)
