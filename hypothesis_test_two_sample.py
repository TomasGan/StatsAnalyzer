import numpy as np
from scipy.stats import norm, t
import math


def hypothesis_testing_diff_means_known_std(sample1_mean, sample1_size, sample2_mean, sample2_size, pop_std1, pop_std2,
                                            delta0=0, alpha=0.05, tail='two'):
    # Calculate the standard error of the difference between the means
    standard_error = np.sqrt((pop_std1 ** 2 / sample1_size) + (pop_std2 ** 2 / sample2_size))

    # Calculate the test statistic
    z = (sample1_mean - sample2_mean - delta0) / standard_error

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
        'sample1_mean': sample1_mean,
        'sample2_mean': sample2_mean,
        'alpha': alpha,
        'tail': tail
    }


def hypothesis_testing_diff_means_unknown_std(data1=None, data2=None, sample1_mean=None, sample1_std=None,
                                              sample1_size=None, sample2_mean=None, sample2_std=None, sample2_size=None,
                                              equal_var=True, delta0=0, alpha=0.05, tail='two'):
    if data1 is not None and data2 is not None:
        sample1_mean = np.mean(data1)
        sample1_std = np.std(data1, ddof=1)
        sample1_size = len(data1)
        sample2_mean = np.mean(data2)
        sample2_std = np.std(data2, ddof=1)
        sample2_size = len(data2)
    elif (sample1_mean is not None and sample1_std is not None and sample1_size is not None and sample2_mean is not None
          and sample2_std is not None and sample2_size is not None):
        pass
    else:
        raise ValueError("Insufficient data provided to perform hypothesis testing.")

    if sample1_std**2 == sample2_std**2:
        # Calculate the pooled standard deviation
        pooled_std = np.sqrt(((sample1_size - 1) * sample1_std ** 2 + (sample2_size - 1) * sample2_std ** 2) / (
                sample1_size + sample2_size - 2))

        # Calculate the standard error of the difference between the means
        standard_error = np.sqrt(1 / sample1_size + 1 / sample2_size) * pooled_std

        # Degrees of freedom
        degrees_of_freedom = sample1_size + sample2_size - 2
    else:
        # Calculate the standard error of the difference between the means
        standard_error = np.sqrt((sample1_std ** 2 / sample1_size) + (sample2_std ** 2 / sample2_size))

        # Degrees of freedom (smaller of n1 - 1 and n2 - 1)
        degrees_of_freedom = min(sample1_size - 1, sample2_size - 1)

    # Calculate the test statistic
    t_statistic = (sample1_mean - sample2_mean - delta0) / standard_error

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
        'degrees_of_freedom': degrees_of_freedom,
        'p_value': p_value,
        'reject_null': reject_null,
        'sample1_mean': sample1_mean,
        'sample2_mean': sample2_mean,
        'alpha': alpha,
        'tail': tail
    }


def hypothesis_testing_paired_samples(data1, data2, alpha=0.05, tail='two'):
    if len(data1) != len(data2):
        raise ValueError("The two samples must have the same length.")

    # Calculate the differences
    differences = np.array(data1) - np.array(data2)

    # Calculate the mean and standard deviation of the differences
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)

    # Calculate the standard error of the mean difference
    standard_error = std_diff / np.sqrt(n)

    # Calculate the test statistic
    t_statistic = mean_diff / standard_error

    # Determine the degrees of freedom
    degrees_of_freedom = n - 1

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
        'degrees_of_freedom': degrees_of_freedom,
        'p_value': p_value,
        'reject_null': reject_null,
        'mean_diff': mean_diff,
        'alpha': alpha,
        'tail': tail
    }


def hypothesis_testing_diff_proportions(sample1_successes, sample1_size, sample2_successes, sample2_size, alpha=0.05,
                                        tail='two'):
    # Calculate sample proportions
    p1_hat = sample1_successes / sample1_size
    p2_hat = sample2_successes / sample2_size

    # Calculate pooled proportion
    pooled_p = (sample1_successes + sample2_successes) / (sample1_size + sample2_size)

    # Calculate the standard error
    standard_error = math.sqrt(pooled_p * (1 - pooled_p) * (1 / sample1_size + 1 / sample2_size))

    # Calculate the test statistic
    z = (p1_hat - p2_hat) / standard_error

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
        'p1_hat': p1_hat,
        'p2_hat': p2_hat,
        'alpha': alpha,
        'tail': tail
    }


# Example 2
# stats = hypothesis_testing_diff_means_known_std(5271, 250, 5121, 250,
#                                                 960, 845, alpha=0.05, tail='two')

# difference between proportions
# Example 1
# stats2 = hypothesis_testing_diff_proportions(sample1_successes=186, sample1_size=200,
#                                              sample2_successes=222, sample2_size=250, alpha=0.10, tail='two')
# Example 2
# stats3 = hypothesis_testing_diff_proportions(28, 42, 12, 37,
#                                              0.01, 'right')

# Chapter Quiz
# P1
# claim: mean score on the reading assessment test for male high school students is less than the mean score for
#        female high school students.
# a)
# Ho: u1 >= u2
# Ha: u1 < u2 claim
# b) left tailed
# z-test, std known, and n1 and n2 > 30
# c) and d)
# q_p1_stats = hypothesis_testing_diff_means_known_std(279, 49, 292, 50,
#                                                      41, 39, 0, 0.05, 'left')
# print(q_p1_stats)
#
# # P2
# # claim: mean scores on a music assessment test for eighth grade students in public and private schools are equal
# # Ho: u1 = u2 claim
# # Ha: u1 =/ u2
# # two tailed
# # t-test, std unknown, and populations are normally distributed and pop variances are equal
# q_p2_stats = hypothesis_testing_diff_means_unknown_std(sample1_size=13, sample1_mean=146, sample1_std=49,
#                                                        sample2_size=15, sample2_mean=160, sample2_std=42, alpha=0.1,
#                                                        tail='two')
# print(q_p2_stats)
#
# # P3
# # claim: the personal finance seminar helps adults increase their credit scores
# # Ho: ud >= 0
# # Ha: ud < 0 claim
# # left tailed
# # t-test because std unknown, samples are random, dependent, and populations are normally distributed
# q_p3_stats = hypothesis_testing_paired_samples([608, 620, 610, 650, 640, 680, 655, 602, 644, 656, 632, 664],
#                                                [646, 692, 715, 669, 725, 786, 700, 650, 660, 650, 680, 702], 0.01,
#                                                'left')
# print(q_p3_stats)
#
# # P4
# # claim: the proportion of U.S. adults who approve of the job the Supreme Court is doing is greater than
# #        it was 3 year ago
# # Ho: p1 <= p2
# # Ha: p1 > p2 claim
# # right tailed
# # z-test for difference of proportions
# q_p4_stats = hypothesis_testing_diff_proportions(584, 1007, 501, 1022,
#                                                  0.05, 'right')
# print(q_p4_stats)
#
# # Chapter Test
# # P1
# # claim: the proportion of students taking the SAT who are undecided on an intended college major has not changed
# # Ho: p1 = p2
# # Ha: p1 =/ p2 claim
# # two tailed
# # z-test for difference of proportions
# t_p1_stats = hypothesis_testing_diff_proportions(350, 5000, 360, 12000,
#                                                  0.10, 'two')
# print(t_p1_stats)
#
# # P2
# # claim: the mean home sales price in Olathe, Kansas, is greater than in Rolla, Missouri
# # Ho: u1 <= u2
# # Ha: u1 > u2 claim
# # right tailed
# # known std z-test for difference of means
# t_p2_stats = hypothesis_testing_diff_means_known_std(sample1_size=39, sample1_mean=392453, pop_std1=224902,
#                                                      sample2_size=38, sample2_mean=285787, pop_std2=330578,
#                                                      alpha=0.05, tail='right')
# print(t_p2_stats)
#
# # P3
# # claim: soft tissue massage therapy helps to reduce the lengths of time patients suffer from headaches
# # Ho: ud <= 0
# # Ha: ud > 0 claim
# # right tailed
# # t-test for difference of means (dependent samples)
# t_p3_stats = hypothesis_testing_paired_samples([5.2, 5.1, 4.9, 1.6, 6.1, 2.3, 4.6, 5.2, 3.1, 4.4, 4.2,
#                                                 5.4, 3.3, 5.2, 3.7, 2.6, 2.7, 2.6],
#                                                [3.5, 3.3, 3.7, 2.3, 2.7, 2.4, 2.1, 2.5, 2.8, 4.1,
#                                                 3.0, 2.4, 2.4, 2.7, 2.6, 2.4, 2.7, 2.4], 0.05, 'right')
# print(t_p3_stats)
#
# # P4
# # claim: the mean household income in a recent year is different for native-born households and foreign-born
# #        households.
# # Ho: u1 = u2
# # Ha: u1 =/ u2 claim
# # two tailed
# # t-test for difference of means (unknown samples), unknown std
# t_p4_stats = hypothesis_testing_diff_means_unknown_std(sample1_size=18, sample1_mean=69474, sample1_std=21249,
#                                                        sample2_size=21, sample2_mean=64900, sample2_std=17896,
#                                                        alpha=0.01, tail='two')
# print(t_p4_stats)