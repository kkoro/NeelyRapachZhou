# Reproduce Table 1 from Neely, Rapach, Zhou (2011)
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
file_name = 'nrz_fundamentals_monthly.pkl'
#file_name = 'fundamentals_monthly.pkl'
fundamentals = pd.read_pickle(path_name + file_name)


# define functions used for calculating summary statistics
def f(x):return Series([x.mean(), x.std(), x.min(), x.max(), x.autocorr()],
                       index=['mean', 'std', 'min', 'max', 'autocorr'])


var_list = ['log_equity_premium', 'dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis',
            'tbl', 'lty', 'ltr', 'tms','dfy','dfr','infl']

# Replicate Table 1, Summary Statistics, from Neely, Rapach, Zhou (2011)
table1 = fundamentals[var_list][beg_date:end_date].apply(f)
pd.options.display.float_format = '{:.2f}'.format
display(table1.transpose())