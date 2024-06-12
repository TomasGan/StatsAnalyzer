import numpy as np
import scipy
import math


# Known standard deviation (σ) and unknown standard deviation
def confidence_interval_mean(data=None, pop_std=None, sample_std=None, sample_mean=None, confidence_level=0.95,
                             sample_size=None):
    if data is not None:
        sample_mean = np.mean(data)
        sample_size = len(data)
        sample_std = np.std(data, ddof=1)
    elif sample_mean is not None and sample_size is not None:
        if sample_std is None and pop_std is None:
            raise ValueError("Sample standard deviation or population standard deviation must be provided.")
    else:
        raise ValueError("Insufficient data provided to compute confidence interval.")

    alpha = 1 - confidence_level

    if pop_std is not None:
        # Using Z-distribution if population standard deviation is known
        z_critical = scipy.stats.norm.ppf(1 - alpha / 2)
        margin_of_error = z_critical * (pop_std / np.sqrt(sample_size))
    else:
        # Using T-distribution if population standard deviation is unknown
        degrees_of_freedom = sample_size - 1
        t_critical = scipy.stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
        margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    print(f"{confidence_level * 100}% Confidence Interval:{lower_bound: .4f} < \u03BC <{upper_bound: .4f}"
          f"\nsample mean: {sample_mean}\n")

    return lower_bound, upper_bound


# Find approximate minimum sample size to estimate population mean
def minimum_sample_estimate(zc=0.0, margin_error=0.0, pop_standard_dev=0.0, level_of_confidence=0.95):
    if margin_error == 0:
        print(f"E cannot equal {margin_error}\n")
        return
    else:
        if zc == 0.0:
            zc = round(scipy.stats.norm.ppf(1 - (1 - level_of_confidence) / 2), 3)
            res = round((zc * pop_standard_dev) / margin_error, 3)
            n = (math.pow(res, 2))
            return math.ceil(n)
        else:
            res = (zc * pop_standard_dev) / margin_error
            n = (math.pow(res, 2))
            return math.ceil(n)


# For population proportions
def confidence_interval_proportion(sample_proportion, sample_size, confidence_level=0.95):
    alpha = 1 - confidence_level
    z_critical = scipy.stats.norm.ppf(1 - alpha / 2)

    margin_of_error = z_critical * math.sqrt((sample_proportion * (1 - sample_proportion)) / sample_size)

    lower_bound = sample_proportion - margin_of_error
    upper_bound = sample_proportion + margin_of_error

    print(f"{confidence_level * 100}% Confidence Interval:{lower_bound: .4f} < p <{upper_bound: .4f}"
          f"\nproportion: {sample_proportion}\n")

    return lower_bound, upper_bound


# Find approximate minimum sample size to estimate population proportion p
def minimum_sample_proportion(point_estimate=0.5, level_of_confidence=0.95, margin_err=0.0, zc=0.0):
    if margin_err == 0.0:
        print(f"Margin of error cannot be {margin_err}")
        return

    q = 1 - point_estimate

    if zc == 0.0:
        zc = scipy.stats.norm.ppf(1 - (1 - level_of_confidence) / 2)

    res = zc / margin_err
    n = point_estimate * q * math.pow(res, 2)

    return math.ceil(n)


def critical_val_x_2(level_of_confidence=0.95, sample_size=0.0):
    df = sample_size - 1
    alpha = 1 - level_of_confidence

    chi2_critical_left = round(scipy.stats.chi2.ppf(alpha / 2, df), 3)
    chi2_critical_right = round(scipy.stats.chi2.ppf(1 - alpha / 2, df), 3)

    return chi2_critical_left, chi2_critical_right


# Confidence intervals for variance and standard population
def confidence_interval_var_std(data=None, sample_variance=None, sample_size=None, level_of_confidence=0.95):
    if data is not None:
        arr = np.array(data)
        n = len(arr)
        sample_variance = np.var(arr, ddof=1)
    elif sample_variance is not None and sample_size is not None:
        n = sample_size
    else:
        raise ValueError("Insufficient data provided to compute confidence interval.")

    xl2, xr2 = critical_val_x_2(level_of_confidence, n)

    lower_interval_variance = ((n - 1) * sample_variance) / xr2
    upper_interval_variance = ((n - 1) * sample_variance) / xl2

    lower_interval_std = math.sqrt(lower_interval_variance)
    upper_interval_std = math.sqrt(upper_interval_variance)

    print(
        f"{level_of_confidence * 100}% Confidence Interval for Variance: {lower_interval_variance:.4f} < \u03C3\u00b2 < "
        f"{upper_interval_variance:.4f}\n")
    print(
        f"{level_of_confidence * 100}% Confidence Interval for Standard Deviation: {lower_interval_std:.4f} < \u03C3 < "
        f"{upper_interval_std:.4f}\n")

    return lower_interval_variance, upper_interval_variance, lower_interval_std, upper_interval_std


# Chapter 6 Quiz

# Q1
confidence_interval_mean([2.42, 2.38, 2.44, 2.67, 2.44, 2.57, 2.39, 2.49, 2.39, 2.41, 2.49,
                          2.40, 2.42, 2.53, 2.39, 2.45, 2.44, 2.54, 2.49, 2.42], pop_std=0.068,
                         sample_size=20, confidence_level=0.95)

# Q2
min_n = minimum_sample_estimate(margin_error=2 / 60, level_of_confidence=0.99, pop_standard_dev=0.068)
print(f"The minimum sample size needed to estimate \u03BC is {min_n}\n")

# Q3 a and b
sample = [7.4, 2.0, 12.1, 8.8, 9.4, 7.3, 1.9, 2.8, 7.0, 7.3]
confidence_interval_mean(sample, confidence_level=0.90)

# Q3 c
confidence_interval_mean(sample, confidence_level=0.90, pop_std=3.5)

# Q4
confidence_interval_mean(sample_mean=133326, sample_std=36729, confidence_level=0.95, sample_size=12)

# Q5
print(f"The t-value, 131935, does fall between t(-0.95) and t(0.95)\n")

# Q6
# a) and b)
confidence_interval_proportion(sample_proportion=838 / 1010, sample_size=1010, confidence_level=0.90)
# c)
print(f"Yes, the values would fall outside of the interval.\n")
# d)
min_n_p = minimum_sample_proportion(level_of_confidence=0.99, margin_err=0.04, point_estimate=838 / 1010)
print(f"The minimum sample size needed to estimate p is {min_n_p}\n")

# Q7
confidence_interval_var_std([7.5, 2.0, 12.1, 8.8, 9.4, 7.3, 1.9, 2.8, 7.0, 7.3], level_of_confidence=.95)

# Chapter 6 Test

# Q1
# a) and b)
confidence_interval_proportion(sample_proportion=1740 / 2096, sample_size=2096, confidence_level=0.95)
# c)
# d)
minimum_sample_proportion(point_estimate=1740 / 2096, level_of_confidence=0.99, margin_err=0.03)

# Q2
q2_test_data = [170, 225, 183, 137, 287, 191, 268, 185, 211, 284]
# a)
print(f"Standard deviation: {np.std(q2_test_data, ddof=1)}\n")
# b)
confidence_interval_mean(data=q2_test_data, sample_size=10, confidence_level=0.95)
# c)
confidence_interval_var_std(data=q2_test_data, level_of_confidence=0.99)

# Q3
q3_test_data = [590, 650, 730, 560, 460, 400, 620, 780, 510, 700, 590, 670]
# a) and b)
confidence_interval_mean(data=q3_test_data, sample_size=12, confidence_level=0.90, pop_std=108)
# c)
# d)
min_n_q3 = minimum_sample_estimate(level_of_confidence=0.95, margin_error=10, pop_standard_dev=108)
print(f"The minimum sample size needed to estimate \u03BC is {min_n_q3}\n")

# Q4
# a) t-distributions because n > 30
confidence_interval_mean(sample_size=40, sample_mean=20, sample_std=7.5, confidence_level=0.95)

# b) standard normal distribution because population is normally distributed
confidence_interval_mean(sample_size=15, sample_mean=11.89, pop_std=0.05, confidence_level=0.90)

# min_sample_size_p = minimum_sample_proportion(level_of_confidence=0.95, margin_err=0.03)
# print(f"The minimum sample size needed to estimate p is {min_sample_size_p}")
# min_sample_size_p = minimum_sample_proportion(level_of_confidence=0.95, margin_err=0.03, point_estimate=0.31)
# print(f"The minimum sample size needed to estimate p is {min_sample_size_p}")

# confidence_interval_pop_prop(level_of_confidence=0.95, point_estimate=0.70, sample_size=540)
# confidence_interval_pop_prop(sample_size=800, level_of_confidence=0.99, point_estimate=0.35)

# min_sample_size = minimum_sample_estimate(1.96, 0.5, 2.4)
# print(f"The minimum sample size needed to estimate \u03BC is {min_sample_size}")

# s_mean, lower, upper = confidence_interval_mean([19, 18, 18, 15, 21, 21, 23, 20, 21, 19, 16,
#                                                  19, 22, 15, 19, 24, 20, 24, 20, 17, 18, 17,
#                                                  19, 20, 20, 20, 22, 24, 22, 23, 23, 21, 22,
#                                                  20, 17, 21, 16, 18, 18, 26], 2.4,
#                                                 0, 0, 0.99)
# s_mean_2, lower_2, upper_2 = confidence_interval_mean(0, 1.5, 22.9,20, .90)
