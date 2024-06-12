import numpy as np
from scipy.stats import chisquare, chi2, chi2_contingency


def goodness_of_fit_test(observed, expected, alpha=0.05):
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    df = len(observed) - 1
    critical_value = chi2.ppf(1 - alpha, df)
    reject_null = chi2_stat > critical_value
    return {
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'critical_value': critical_value,
        'reject_null': reject_null
    }


def chi_square_test_independence(observed, alpha=0.05):
    chi2_stat, p_value, dof, expected = chi2_contingency(observed)
    critical_value = chi2.ppf(1 - alpha, dof)
    reject_null = chi2_stat > critical_value
    return {
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'critical_value': critical_value,
        'reject_null': reject_null,
        'expected': expected
    }