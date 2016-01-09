from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Note: currently only handles univariate x

def oos_forecast(y, x, df, beg_date_init, beg_date_oos, end_date_oos):

    oos_results = DataFrame(columns=['y_forecast', 'res', 'obs'],
                            index=df[beg_date_oos:end_date_oos].index)
    # x variables are already lagged
    formula = y + ' ~ ' + x

    #df = df[beg_date_oos:end_date_oos]
    df = df[beg_date_init:end_date_oos]

    for curr_date in df[beg_date_oos:end_date_oos].index:
        # run bivariate regression for each out-of-sample period
        df_sub = df[df.index < curr_date]
        results = smf.ols(formula, data=df_sub).fit()
        oos_results['obs'][curr_date] = len(df_sub)

        # use estimated coefficient and curr_date x to forecast y
        oos_results['y_forecast'][curr_date] = results.params['Intercept'] + \
            results.params[x] * df[x][curr_date]
        oos_results['res'][curr_date] = oos_results['y_forecast'][curr_date] - \
             df[y][curr_date]

    return oos_results