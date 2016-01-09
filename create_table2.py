# Reproduce Table 2 from Neely, Rapach, Zhou (2011)
#
# Larry Takeuchi
# first version: 11/20/2015
#
# If necessary, change director in iPython:
#  import os
#  os.chdir('/Users/ltakeuchi/Python/stockpredict')
 

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
#import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# set date range (1950-12 to 2011-12 for NRZ)
beg_date = '1950-12'
end_date = '2011-12'

# load data
path_name = '/Users/ltakeuchi/Python/stockpredict/data/'
file_name = 'nrz_fundamentals_monthly.pkl'
#file_name = 'fundamentals_monthly.pkl'
df = pd.read_pickle(path_name + file_name)

file_name = 'nrz_technical_variables.pkl'
tech = pd.read_pickle(path_name + file_name)

# load NBER recession indicator
nber = pd.read_pickle(path_name + 'nber_monthly.pkl')

# add recession dummies to df
df = pd.concat([df, nber], axis=1)
df = pd.concat([df, tech], axis=1)

# save combined econ+tech dataset
#df.to_pickle(path_name + 'nrz_all_monthly.pkl')


# change sign so that expected slope coefficents are positive as in NRZ
df['ntis'] = df['ntis'] * -1
df['tbl'] = df['tbl'] * -1
df['lty'] = df['lty'] * -1
df['infl'] = df['infl'] * -1

econ_var = ['dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr',
            'tms','dfy','dfr','infl']
tech_var = ['ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12',
            'mom_9', 'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12',
            'vol_3_9', 'vol_3_12']
var_list = econ_var + tech_var


# run bivariate forecasting regression y(t) on x(t-1) for t = 1951-01 to 2011-12
# TO-DO: p-values (from wild bootstrap)

# get data for specified date range
df = df[beg_date:end_date]

#------------------------------------------------------------------------------
# calculate principal components with standardized data

# economic
df_sub = df[econ_var].copy()
df_sub = df_sub.shift(1)
df_sub_std = StandardScaler().fit_transform(df_sub[df_sub.index[0]+1:df_sub.index[-1]])
pca_model = PCA(n_components=3)
pca_econ = pca_model.fit_transform(df_sub_std)
dates = pd.date_range(start=df_sub.index[0]+1, end=df_sub.index[-1], freq='M')
pca_econ = DataFrame(data=pca_econ, index=dates, columns=['econ1', 'econ2', 'econ3'])

# tech
df_sub = df[tech_var].copy()
df_sub = df_sub.shift(1)
df_sub_std = StandardScaler().fit_transform(df_sub[df_sub.index[0]+1:df_sub.index[-1]])
pca_model = PCA(n_components=1)
pca_tech = pca_model.fit_transform(df_sub_std)
dates = pd.date_range(start=df_sub.index[0]+1, end=df_sub.index[-1], freq='M')
pca_tech = DataFrame(data=pca_tech, index=dates, columns=['tech1'])

# all (econ + tech)
df_sub = df[var_list].copy()
df_sub = df_sub.shift(1)
df_sub_std = StandardScaler().fit_transform(df_sub[df_sub.index[0]+1:df_sub.index[-1]])
pca_model = PCA(n_components=4)
pca_all = pca_model.fit_transform(df_sub_std)
dates = pd.date_range(start=df_sub.index[0]+1, end=df_sub.index[-1], freq='M')
pca_all = DataFrame(data=pca_all, index=dates, columns=['all1', 'all2', 'all3', 'all4'])

#------------------------------------------------------------------------------

# lag the x variables
df[var_list] = df[var_list].shift(1)

# initialize dictionary of lists
d = {}
for i in ['coeff', 't-stat', 'r2', 'r2_exp', 'r2_rec']:
    d[i] = []

# define functions used for r-squared for subperiod
def calc_subperiod_r2(y, res, indicator):
    ss_total = sum(np.power(np.multiply(y, indicator) - np.mean(y) * indicator, 2))
    ss_residual = sum(np.power(np.multiply(res, indicator), 2))
    return (1 - ss_residual / ss_total)

# run bivariate regression for each predictor variable
y_var = 'log_equity_premium'
for x_var in var_list:
    formula = y_var + ' ~ ' + x_var
    results = smf.ols(formula, data=df).fit()
   
    d['coeff'].append(results.params[x_var])
    d['t-stat'].append(results.params[x_var] / results.HC0_se[x_var])
    d['r2'].append(results.rsquared * 100)
    d['r2_exp'].append(calc_subperiod_r2(df[y_var][1:], results.resid,
                       1 - df['recession'][1:]) * 100)
    d['r2_rec'].append(calc_subperiod_r2(df[y_var][1:], results.resid,
                       df['recession'][1:]) * 100)


# create and display Table 2, panel A
pd.options.display.float_format = '{:.2f}'.format
table2a = DataFrame(d, index=var_list)
table2a = table2a[['coeff', 't-stat', 'r2', 'r2_exp', 'r2_rec']]
display(table2a)


# run multivariate regression with principal components
econ_pc = ['econ1', 'econ2', 'econ3']
tech_pc = ['tech1']
all_pc = ['all1', 'all2', 'all3', 'all4']
listoflists = [econ_pc, tech_pc, all_pc]

pca_data = pd.concat([df[['log_equity_premium', 'recession']], pca_econ,
    pca_tech, pca_all], axis=1)

for var_list in listoflists:
    formula = 'log_equity_premium ~ ' + ' + '.join(var_list)
    pca_results = smf.ols(formula, data=pca_data).fit()

    # Table 2, panel B
    table2b = DataFrame(columns=('coeff', 't-stat', 'r2', 'r2_exp', 'r2_rec'), index=var_list)
    for x_var in var_list:
        table2b.loc[x_var, 'coeff'] = pca_results.params[x_var]
        table2b.loc[x_var, 't-stat'] = pca_results.params[x_var] / pca_results.HC0_se[x_var]

    table2b.loc[var_list[0], 'r2'] = pca_results.rsquared * 100
    table2b.loc[var_list[0], 'r2_exp'] = calc_subperiod_r2(pca_data[y_var][1:],
        pca_results.resid, 1 - df['recession'][1:]) * 100
    table2b.loc[var_list[0], 'r2_rec'] = calc_subperiod_r2(pca_data[y_var][1:],
        pca_results.resid, df['recession'][1:]) * 100
    display(table2b)


