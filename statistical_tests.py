from numpy import mean
from numpy import sqrt
import numpy as np
from scipy.stats import sem
from scipy.stats import t
from scipy import stats


def independent_t_test(data1, data2, alpha):
    mean1, mean2 = mean(data1), mean(data2)
    # standard error
    se1, se2 = sem(data1), sem(data2)
    # squared error on difference between samples
    sed = sqrt(se1**2.0 + se2**2.0)
    # calculate t-statistic
    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
    df = len(data1) + len(data2) - 2
    # calculate critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0

    if alpha > p:
        print('Reject')
    else:
        print('Accept')
    return t_stat, df, cv, p


def welch_t_test(data1, data2):
    """
    t, df, p = welch_t_test(data1, data2)
    """
    n1 = data1.size
    n2 = data2.size
    mu1 = np.mean(data1)
    mu2 = np.mean(data2)
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=2)
    t = (mu1 - mu2) / np.sqrt(var1 / n1 + var2 / n2)
    df = (var1 / n1 + var2 / n2)**2 / (var1**2 / (n1**2 * (n1 - 1)) + var2**2 / (n2**2 * (n2 - 1)))
    p = 2 * stats.t.cdf(-abs(t), df)
    return t, df, p


def p_value_interpreter(p, alpha):
    """ Usage:
    p_value_interpreter(p, alpha)
    """
    # interpret via p-value
    if p > alpha:
        print('Accept null hypothesis that the means are equal.')
    else:
        print('Reject the null hypothesis that the means are equal.')
