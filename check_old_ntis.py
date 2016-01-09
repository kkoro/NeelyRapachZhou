# Import Goyal stock prediction data from website
#
# Larry Takeuchi
# first version: 11/18/2015

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import display, HTML

path_name = '/Users/ltakeuchi/NeelyRapachZhou2014/raw_data/'
file_name = 'Returns_econ_tech_data.xls'

goyal = pd.ExcelFile(path_name + file_name)
#dfs = {sheet_name: goyal.parse(sheet_name, parse_dates=[0]) for sheet_name in goyal.sheet_names}
dfs = {sheet_name: goyal.parse(sheet_name) for sheet_name in goyal.sheet_names}


# create date index for monthly
dfs['Monthly']['Date'] = pd.to_datetime(dfs['Monthly']['Date'].astype(str), format='%Y%m')
dfs['Monthly'].set_index('Date', inplace=True)
dfs['Monthly'].index = dfs['Monthly'].index.to_period('M').to_timestamp('M')
df = dfs['Monthly']

df.rename(columns={'CRSP value-weighted S&P 500 return':'CRSP_SPvw'}, inplace=True)
df.rename(columns={'Risk-free rate':'Rfree'}, inplace=True)
df.rename(columns={'Net equity expansion':'ntis'}, inplace=True)
df.rename(columns={'Inflation rate':'infl'}, inplace=True)

df['Rfree'] = df['Rfree'].shift(1)
df['log_equity_premium'] = (np.log1p(df['CRSP_SPvw']) - np.log1p(df['Rfree'])) * 100
df['infl'] = df['infl'].shift(1) * 100

# set date range
beg_date = '1950-12'
end_date = '2011-12'

var_list = ['ntis', 'infl']

# run bivariate forecasting regression y(t) on x(t-1) for t = 1951-01 to 2011-12

# slope coefficient
# Newey-West t-statistic
# p-values (from wild bootstrap)
# R2, R2 expansion, R2 recession

# get data for specified date range
df = df[beg_date:end_date]

# lag the x variables
df[var_list] = df[var_list].shift(1)

# initialize dictionary of lists
d = {}
d['coeff'] = []
d['t-stat'] = []
d['rsquared'] = []

# run bivariate regression for each predictor variable
for x_var in var_list:
    formula = 'log_equity_premium ~' + x_var
    results = smf.ols(formula, data=df).fit()
   
    d['coeff'].append(results.params[x_var])
    d['t-stat'].append(results.params[x_var] / results.HC0_se[x_var])
    d['rsquared'].append(results.rsquared * 100)

# create and display Table 2
table2 = DataFrame(d, index=var_list)
table2 = table2[['coeff', 't-stat', 'rsquared']]
display(table2)