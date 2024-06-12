import numpy as np
from scipy.stats import f, f_oneway


def f_test_two_sample_variance(data1, data2, alpha=0.05, tail='two'):
    n1 = len(data1)
    n2 = len(data2)
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)

    f_stat = var1 / var2 if var1 > var2 else var2 / var1
    dfn = n1 - 1 if var1 > var2 else n2 - 1
    dfd = n2 - 1 if var1 > var2 else n1 - 1

    if tail == 'two':
        critical_value_low = f.ppf(alpha / 2, dfd, dfn)
        critical_value_high = f.ppf(1 - alpha / 2, dfd, dfn)
        reject_null = f_stat < critical_value_low or f_stat > critical_value_high
        critical_values = (critical_value_low, critical_value_high)
    elif tail == 'left':
        critical_value = f.ppf(alpha, dfd, dfn)
        reject_null = f_stat < critical_value
        critical_values = (critical_value,)
    elif tail == 'right':
        critical_value = f.ppf(1 - alpha, dfd, dfn)
        reject_null = f_stat > critical_value
        critical_values = (critical_value,)
    else:
        raise ValueError("Invalid value for 'tail'. Choose from 'two', 'left', 'right'.")

    p_value = 1 - f.cdf(f_stat, dfn, dfd) if tail in ['left', 'two'] else f.cdf(f_stat, dfn, dfd)

    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'dfn': dfn,
        'dfd': dfd,
        'reject_null': reject_null,
        'critical_values': critical_values
    }


def f_test_two_sample_variance_from_stats(sample1_variance, sample1_size, sample2_variance,
                                          sample2_size, alpha=0.05, tail='two'):
    f_stat = sample1_variance / sample2_variance if (
            sample1_variance > sample2_variance) else (
            sample2_variance / sample1_variance)
    dfn = sample1_size - 1 if sample1_variance > sample2_variance else sample2_size - 1
    dfd = sample2_size - 1 if sample1_variance > sample2_variance else sample1_size - 1

    if tail == 'two':
        critical_value_low = f.ppf(alpha / 2, dfd, dfn)
        critical_value_high = f.ppf(1 - alpha / 2, dfd, dfn)
        reject_null = f_stat < critical_value_low or f_stat > critical_value_high
        critical_values = (critical_value_low, critical_value_high)
    elif tail == 'left':
        critical_value = f.ppf(alpha, dfd, dfn)
        reject_null = f_stat < critical_value
        critical_values = (critical_value,)
    elif tail == 'right':
        critical_value = f.ppf(1 - alpha, dfd, dfn)
        reject_null = f_stat > critical_value
        critical_values = (critical_value,)
    else:
        raise ValueError("Invalid value for 'tail'. Choose from 'two', 'left', 'right'.")

    p_value = 1 - f.cdf(f_stat, dfn, dfd) if tail in ['left', 'two'] else f.cdf(f_stat, dfn, dfd)

    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'dfn': dfn,
        'dfd': dfd,
        'reject_null': reject_null,
        'critical_values': critical_values
    }


def one_way_anova(alpha, *groups):
    f_stat, p_value = f_oneway(*groups)
    k = len(groups)
    n = sum(len(group) for group in groups)
    df_between = k - 1
    df_within = n - k

    critical_value = f.ppf(1 - alpha, df_between, df_within)
    reject_null = f_stat > critical_value

    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'df_between': df_between,
        'df_within': df_within,
        'critical_value': critical_value,
        'reject_null': reject_null
    }