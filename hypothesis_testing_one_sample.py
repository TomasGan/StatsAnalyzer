import numpy as np
from scipy.stats import norm, t, chi2
import math


def hypothesis_testing_mean_known_std(data=None, sample_mean=None, sample_size=None, pop_mean=None, pop_std=None,
                                      alpha=0.05, tail='two'):
    if data is not None and len(data) != 0:
        sample_mean = np.mean(data)
        sample_size = len(data)
    elif sample_mean is not None and sample_size is not None:
        pass
    else:
        raise ValueError("Insufficient data provided to perform hypothesis testing.")

    if pop_std is None:
        raise ValueError("Population standard deviation must be provided.")

    # Calculate the test statistic
    z = (sample_mean - pop_mean) / (pop_std / np.sqrt(sample_size))

    # Determine the critical value and decision rule based on the tail type
    if tail == 'two':
        z_critical = norm.ppf(1 - alpha / 2)
        p_value = 2 * (1 - norm.cdf(abs(z)))
        reject_null = abs(z) > z_critical
    elif tail == 'left':
        z_critical = norm.ppf(alpha)
        p_value = norm.cdf(z)
        reject_null = z < z_critical
    elif tail == 'right':
        z_critical = norm.ppf(1 - alpha)
        p_value = 1 - norm.cdf(z)
        reject_null = z > z_critical
    else:
        raise ValueError("Invalid value for 'tail'. Choose from 'two', 'left', 'right'.")

    return {
        'z': z,
        'z_critical': z_critical,
        'p_value': p_value,
        'reject_null': reject_null,
        'sample_mean': sample_mean,
        'pop_mean': pop_mean,
        'alpha': alpha,
        'tail': tail
    }


def hypothesis_testing_mean_unknown_std(data=None, sample_mean=None, sample_size=None, pop_mean=None, sample_std=None,
                                        alpha=0.05, tail='two'):
    if data is not None and len(data) != 0:
        sample_mean = np.mean(data)
        sample_size = len(data)
        sample_std = np.std(data, ddof=1)
    elif sample_mean is not None and sample_size is not None and sample_std is not None:
        pass
    else:
        raise ValueError("Insufficient data provided to perform hypothesis testing.")

    # Calculate the test statistic
    t_statistic = (sample_mean - pop_mean) / (sample_std / np.sqrt(sample_size))

    # Determine the degrees of freedom
    degrees_of_freedom = sample_size - 1

    # Determine the critical value and decision rule based on the tail type
    if tail == 'two':
        t_critical = t.ppf(1 - alpha / 2, degrees_of_freedom)
        p_value = 2 * (1 - t.cdf(abs(t_statistic), degrees_of_freedom))
        reject_null = abs(t_statistic) > t_critical
    elif tail == 'left':
        t_critical = t.ppf(alpha, degrees_of_freedom)
        p_value = t.cdf(t_statistic, degrees_of_freedom)
        reject_null = t_statistic < t_critical
    elif tail == 'right':
        t_critical = t.ppf(1 - alpha, degrees_of_freedom)
        p_value = 1 - t.cdf(t_statistic, degrees_of_freedom)
        reject_null = t_statistic > t_critical
    else:
        raise ValueError("Invalid value for 'tail'. Choose from 'two', 'left', 'right'.")

    return {
        't': t_statistic,
        't_critical': t_critical,
        'p_value': p_value,
        'degrees_of_freedom': degrees_of_freedom,
        'reject_null': reject_null,
        'sample_mean': sample_mean,
        'pop_mean': pop_mean,
        'alpha': alpha,
        'tail': tail
    }


def hypothesis_testing_proportion(sample_proportion, sample_size, pop_proportion, alpha=0.05, tail='two'):
    # Calculate the test statistic
    p_hat = sample_proportion
    p = pop_proportion
    n = sample_size

    standard_error = math.sqrt((p * (1 - p)) / n)
    z = (p_hat - p) / standard_error

    # Determine the critical value and decision rule based on the tail type
    if tail == 'two':
        z_critical = norm.ppf(1 - alpha / 2)
        p_value = 2 * (1 - norm.cdf(abs(z)))
        reject_null = abs(z) > z_critical
    elif tail == 'left':
        z_critical = norm.ppf(alpha)
        p_value = norm.cdf(z)
        reject_null = z < z_critical
    elif tail == 'right':
        z_critical = norm.ppf(1 - alpha)
        p_value = 1 - norm.cdf(z)
        reject_null = z > z_critical
    else:
        raise ValueError("Invalid value for 'tail'. Choose from 'two', 'left', 'right'.")

    return {
        'z': z,
        'z_critical': z_critical,
        'p_value': p_value,
        'reject_null': reject_null,
        'sample_proportion': sample_proportion,
        'pop_proportion': pop_proportion,
        'alpha': alpha,
        'tail': tail
    }


def hypothesis_testing_variance(data=None, sample_variance=None, sample_size=None, pop_variance=None, alpha=0.05,
                                tail='two'):
    if data is not None and len(data) != 0:
        sample_size = len(data)
        sample_variance = np.var(data, ddof=1)
    elif sample_variance is not None and sample_size is not None:
        pass
    else:
        raise ValueError("Insufficient data provided to perform hypothesis testing.")

    if pop_variance is None:
        raise ValueError("Population variance must be provided.")

    # Calculate the test statistic
    chi2_statistic = (sample_size - 1) * sample_variance / pop_variance

    # Determine the degrees of freedom
    degrees_of_freedom = sample_size - 1

    alphaR = alpha / 2
    alphaL = 1 - alpha / 2

    # Determine the critical value and decision rule based on the tail type
    if tail == 'two':
        chi2_critical_low = chi2.ppf(alphaR, df=degrees_of_freedom)
        chi2_critical_high = chi2.ppf(alphaL, df=degrees_of_freedom)
        p_value = 2 * min(chi2.cdf(chi2_statistic, degrees_of_freedom),
                          1 - chi2.cdf(chi2_statistic, degrees_of_freedom))
        reject_null = chi2_statistic < chi2_critical_low or chi2_statistic > chi2_critical_high
    elif tail == 'left':
        chi2_critical = chi2.ppf(alpha, degrees_of_freedom)
        p_value = chi2.cdf(chi2_statistic, degrees_of_freedom)
        reject_null = chi2_statistic < chi2_critical
    elif tail == 'right':
        chi2_critical = chi2.ppf(1 - alpha, degrees_of_freedom)
        p_value = 1 - chi2.cdf(chi2_statistic, degrees_of_freedom)
        reject_null = chi2_statistic > chi2_critical
    else:
        raise ValueError("Invalid value for 'tail'. Choose from 'two', 'left', 'right'.")

    return {
        'chi2_statistic': chi2_statistic,
        'chi2_critical_low': chi2_critical_low if tail == 'two' else None,
        'chi2_critical_high': chi2_critical_high if tail == 'two' else chi2_critical,
        'degrees_of_freedom': degrees_of_freedom,
        'p_value': p_value,
        'reject_null': reject_null,
        'sample_variance': sample_variance,
        'pop_variance': pop_variance,
        'alpha': alpha,
        'tail': tail
    }


# chapter_7_quiz_q6_stat = hypothesis_testing_variance(sample_variance=118.81, sample_size=35, pop_variance=102.01,
#                                                      alpha=0.10, tail='two')
# print(chapter_7_quiz_q6_stat)