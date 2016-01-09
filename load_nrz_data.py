# Import Neely, Rapach, Zhou data
#
# Larry Takeuchi
# first version: 11/21/2015

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
from IPython.display import display

# load NRZ dataset
path_name = '/Users/ltakeuchi/Python/stockpredict/data/'
file_name = 'Returns_econ_tech_data.xls'

# date range to match Neely, Rapach, Zhou (2011)
beg_date = '1950-12'
end_date = '2011-12'

goyal = pd.ExcelFile(path_name + file_name)
dfs = {sheet_name: goyal.parse(sheet_name) for sheet_name in goyal.sheet_names}

# create date index for monthly
dfs['Monthly']['Date'] = pd.to_datetime(dfs['Monthly']['Date'].astype(str), format='%Y%m')
dfs['Monthly'].set_index('Date', inplace=True)
dfs['Monthly'].index = dfs['Monthly'].index.to_period('M').to_timestamp('M')
df = dfs['Monthly']

df.columns = ['spindex', 'D12', 'E12', 'bm', 'tbl', 'AAA', 'BAA', 'lty',
              'ntis', 'Rfree', 'infl', 'ltr', 'corpr', 'svar', 'CRSP_SPvw',
              'CRSP_SPvwx', 'vol']
# dividend payout ratio (log)
df['de'] = np.log(df['D12']) - np.log(df['E12'])


# Rfree is end-of-month tbl divided by 12 (so applies to the next month)
df['Rfree'] = df['Rfree'].shift(1)

# equity premium and log equity premium
df['equity_premium'] = (df['CRSP_SPvw'] - df['Rfree']) * 100
df['log_equity_premium'] = (np.log1p(df['CRSP_SPvw']) - np.log1p(df['Rfree'])) * 100

# dividend-price ratio (log)
df['dp'] = np.log(df['D12']) - np.log(df['spindex'])

# divdend yield (log)
df['dy'] = np.log(df['D12']) - np.log(df['spindex'].shift(1))

# earnings-price ratio (log)
df['ep'] = np.log(df['E12']) - np.log(df['spindex'])

# Treasury bill rate (annual %)
df['tbl'] = df['tbl'] * 100

# Long-term yield (annual %)
df['lty'] = df['lty'] * 100

# Long-term return (%)
df['ltr'] = df['ltr'] * 100

# Term spread (annual %)
df['tms'] = df['lty'] - df['tbl']

# Default yield spread
df['dfy'] = (df['BAA'] - df['AAA']) * 100

# Default return spread
df['dfr'] = df['corpr'] * 100 - df['ltr']

# Inflation (%, lagged)
# Note: inflation series in Goyal 2014 different from earlier versions 
df['infl'] = df['infl'].shift(1) * 100

# Equity risk premium volatility (Mele 2007, JFE)
df['rvol'] = pd.rolling_sum(np.absolute(df['equity_premium']/100), 12) * np.sqrt(np.pi / 2) * np.sqrt(12) / 12


fundamentals = df[['log_equity_premium', 'dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty',
                 'ltr','tms','dfy','dfr','infl', 'equity_premium', 'Rfree']]
technicals = df[['spindex', 'vol']]

# Save dataset to file
fundamentals.to_pickle(path_name + 'nrz_fundamentals_monthly.pkl')
technicals.to_pickle(path_name + 'nrz_technicals_monthly.pkl')

# define functions used for calculating summary statistics
def f(x):return Series([x.mean(), x.std(), x.min(), x.max(), x.autocorr()],
                       index=['mean', 'std', 'min', 'max', 'autocorr'])


# Replicate Table 1, Summary Statistics, from Neely, Rapach, Zhou (2011)
table1 = fundamentals[beg_date:end_date].apply(f)
display(table1.transpose())
