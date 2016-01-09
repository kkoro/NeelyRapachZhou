from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython.display import display, HTML

def perform_selection(y, x, df):
    """
    Select the number of principal components that maximizes adjusted R-squared.
    """
    T = len(x)
    r2 = []

    for k in range(0, T):
        formula = y + ' ~ ' + ' + '.join(x[0:(k+1)])
        #print(formula)
        results = smf.ols(formula, data=df).fit()
        r2.append(results.rsquared_adj)

    return r2.index(max(r2)) + 1



def oos_pc_forecast(y, x, df, max_k, beg_date_init, beg_date_oos, end_date_oos):
    """
    Inputs:
     y = name of dependent variable 
     x = list of lagged predictor variable(s) from which to extract pc        
     df = dataframe containing y and x variables
     max_k = maximum number of principal components to use for forecast
     beg_date_init = starting date for sample 
     beg_date_oos = starting date for forecasts
     end_date_oos = ending date for forecasts

    Output:
     y_forecast = vector of OOS forecasts using principal components from x
     with number of components selected to maximize R-squared in past data
     res = vector of residuals (y_forecast - y_actual)
     obs = vector of number of observation used in forecast regression

    """
    oos_results = DataFrame(columns=['y_forecast', 'res', 'obs'],
                            index=df[beg_date_oos:end_date_oos].index)
    # x variables are already lagged
    #formula = y + ' ~ ' + x

    #df = df[beg_date_oos:end_date_oos]
    df = df[beg_date_init:end_date_oos]


    for curr_date in df[beg_date_oos:end_date_oos].index:

        # run bivariate regression for each out-of-sample period
        df_sub = df[df.index <= curr_date].copy() 

        # get principal components from x variables
        pca_model = PCA(n_components=max_k)
        df_sub_std = StandardScaler().fit_transform(df_sub[x]) 
        pca_results = pca_model.fit_transform(df_sub_std)
        dates = df_sub.index

        pc_list = []
        for i in range(0, max_k):
            pc_list.append('pc' + str(i+1))
        pca_sub = DataFrame(data=pca_results, index=dates, columns=pc_list)

        # add pca factors to df_sub
        df_sub = pd.concat([df_sub, pca_sub], axis=1)

        #  shorten sample by one month to select number of factors
        df_sub_short = df_sub[df_sub.index < curr_date]
        # determine number of PC factors to use
        best_k = perform_selection(y, pca_sub.columns, df_sub_short)
        
        # regresion model
        var_list = []
        for i in range(0, best_k):
            var_list.append('pc' + str(i+1))

        formula = y + ' ~ ' + ' + '.join(var_list)

        
        # use df_sub (not df_sub_short)
        results = smf.ols(formula, data=df_sub_short).fit()

        oos_results['obs'][curr_date] = len(df_sub_short)
        oos_results['y_forecast'][curr_date] = results.params['Intercept'] + \
            np.asscalar(np.inner(pca_sub[pca_sub.index==curr_date][var_list],
            results.params[var_list]))
        oos_results['res'][curr_date] = oos_results['y_forecast'][curr_date] - \
            df_sub[y][curr_date]

    return oos_results



