# Import Goyal stock prediction data from website
#
# Larry Takeuchi
# first version: 11/18/2015

from pandas import Series, DataFrame
import pandas as pd

path_name = '/Users/ltakeuchi/Python/stockpredict/data/'
file_name = 'NBER_recessions.xlsx'

nber = pd.ExcelFile(path_name + file_name)
dfs = nber.parse('Monthly')
dfs['Date'] = pd.to_datetime(dfs['Date'].astype(str), format='%Y%m')

dfs.set_index('Date', inplace=True)
dfs.index = dfs.index.to_period('M').to_timestamp('M')
dfs.to_pickle(path_name + 'nber_monthly.pkl')







