import numpy as np
from scipy.stats import norm

def perform_cw_test(actual, restricted, unrestricted):
# Performs the Clark-West (2007) test to compare forecasts from nested models.
# 
# Input:
#   actual = vector of actual values
#   restricted = vector of forecasts from restricted model
#   unrestricted = vector of forecasts from untrestricted model
#
# Output:
#   MSPE_adj = Clark and West statistic
#   p_value = correponding p-value
#
# Reference: T.E. Clark and K.D. West (2007). "Approximately Normal Tests
# for Equal Predictive Accuracy in Nested Models." Journal of
# Econometrics 138, 291-311

    e1 = actual - restricted
    e2 = actual - unrestricted
    f_hat = np.power(e1, 2) - (np.power(e2, 2) - np.power(restricted - unrestricted, 2))
    Y_f = f_hat
    X_f = np.ones((len(Y_f), 1))

    beta_f = np.dot(np.linalg.inv(np.dot(np.transpose(X_f), X_f)), (np.dot(np.transpose(X_f), Y_f)))
    e_f = Y_f - np.dot(X_f, beta_f)
    sig2_e = np.dot(e_f, e_f) / (len(Y_f) - 1)
    cov_beta_f = np.dot(sig2_e, np.linalg.inv(np.dot(np.transpose(X_f), X_f)))
    MSPE_adj = beta_f / np.sqrt(cov_beta_f)
    p_value = 1 - norm.cdf(MSPE_adj[(0,0)])
    
    return {'MSPE_adj':MSPE_adj[(0,0)], 'p_value':p_value}