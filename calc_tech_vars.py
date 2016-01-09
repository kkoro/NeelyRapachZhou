# Calculate technical variables from Neely, Rapach, Zhou (2011)
#
# Larry Takeuchi
# first version: 11/21/2015

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from IPython.display import display

# date range to match Neely, Rapach, Zhou (2011)
beg_date = '1950-12'
end_date = '2011-12'

# load data
path_name = '/Users/ltakeuchi/Python/stockpredict/data/'
file_name = 'nrz_technicals_monthly.pkl'
df = pd.read_pickle(path_name + file_name)

# calculate technical variables used in NRZ

# moving average indicators
df['ma1'] = df['spindex']
df['ma2'] = pd.rolling_mean(df['spindex'], 2, 2)
df['ma3'] = pd.rolling_mean(df['spindex'], 3, 3)
df['ma9'] = pd.rolling_mean(df['spindex'], 9, 9)
df['ma12'] = pd.rolling_mean(df['spindex'], 12, 12)
df['ma_1_9'] = np.where(df['ma9'].isnull(), np.nan,
	(df['ma1'] > df['ma9']).astype(int))
df['ma_1_12'] = np.where(df['ma12'].isnull(), np.nan,
	(df['ma1'] > df['ma12']).astype(int))
df['ma_2_9'] = np.where(df['ma9'].isnull(), np.nan,
	(df['ma2'] > df['ma9']).astype(int))
df['ma_2_12'] = np.where(df['ma12'].isnull(), np.nan,
	(df['ma2'] > df['ma12']).astype(int))
df['ma_3_9'] = np.where(df['ma9'].isnull(), np.nan,
	(df['ma3'] > df['ma9']).astype(int))
df['ma_3_12'] = np.where(df['ma12'].isnull(), np.nan,
	(df['ma3'] > df['ma12']).astype(int))

# momentum indicators
df['lag9'] = df['spindex'].shift(9)
df['mom_9'] = np.where(df['lag9'].isnull(), np.nan,
	(df['spindex'] >= df['lag9']).astype(int))

df['lag12'] = df['spindex'].shift(12)
df['mom_12'] = np.where(df['lag12'].isnull(), np.nan,
	(df['spindex'] >= df['lag12']).astype(int))

# volume indicators

# on-balance volume (obv)
df['obv'] = (df['spindex'] >= df['spindex'].shift(1)).astype(int) * 2  - 1
df['obv'] = df['obv'] * df['vol']
df['obv'] = df['obv'].cumsum()

df['obv_ma1'] = df['obv']
df['obv_ma2'] = pd.rolling_mean(df['obv'], 2, 2)
df['obv_ma3'] = pd.rolling_mean(df['obv'], 3, 3)
df['obv_ma9'] = pd.rolling_mean(df['obv'], 9, 9)
df['obv_ma12'] = pd.rolling_mean(df['obv'], 12, 12)

# Need to set obs before start_vol to nan
df['valid'] = np.where(df['obv'].isnull(), np.nan, 1)

df['vol_1_9'] = (df['obv_ma1'] >= df['obv_ma9']).astype(int) * df['valid']
df['vol_1_12'] = (df['obv_ma1'] >= df['obv_ma12']).astype(int) * df['valid']
df['vol_2_9'] = (df['obv_ma2'] >= df['obv_ma9']).astype(int) * df['valid']
df['vol_2_12'] = (df['obv_ma2'] >= df['obv_ma12']).astype(int) * df['valid']
df['vol_3_9'] = (df['obv_ma3'] >= df['obv_ma9']).astype(int) * df['valid']
df['vol_3_12'] = (df['obv_ma3'] >= df['obv_ma12']).astype(int) * df['valid']

#define functions used for calculating summary statistics
def f(x):return Series([x.mean(), x.std(), x.min(), x.max(), x.autocorr()],
                       index=['mean', 'std', 'min', 'max', 'autocorr'])

var_list = ['ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12',
            'mom_9', 'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12',
            'vol_3_9', 'vol_3_12']

# Replicate Table 1, Summary Statistics, from Neely, Rapach, Zhou (2011)
table1 = df[var_list][beg_date:end_date].apply(f)
display(table1.transpose())

#print(df[var_list])
# Save dataset to file
df[var_list].to_pickle(path_name + 'nrz_technical_variables.pkl')
