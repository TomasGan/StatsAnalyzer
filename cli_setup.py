import argparse
from hypothesis_testing_one_sample import (
    hypothesis_testing_variance, hypothesis_testing_proportion,
    hypothesis_testing_mean_known_std, hypothesis_testing_mean_unknown_std)
from hypothesis_test_two_sample import (
    hypothesis_testing_paired_samples, hypothesis_testing_diff_proportions,
    hypothesis_testing_diff_means_known_std,
    hypothesis_testing_diff_means_unknown_std)


def prompt_for_test_parameters(test_type):
    params = {}

    if test_type == 'z_test_mean':
        params['pop_mean'] = float(input("Enter the population mean: "))
        params['pop_std'] = float(input("Enter the population standard deviation: "))
        params['alpha'] = float(input("Enter the significance level (default: 0.05): ") or 0.05)
        params['tail'] = input("Enter the type of test ('two', 'left', 'right', default: 'two'): ") or 'two'

        data_input = input("Enter the sample data separated by spaces or press Enter to provide sample mean and size: ")
        if data_input:
            params['data'] = list(map(float, data_input.split()))
        else:
            params['sample_mean'] = float(input("Enter the sample mean: "))
            params['sample_size'] = int(input("Enter the sample size: "))

    elif test_type == 't_test_mean':
        params['pop_mean'] = float(input("Enter the population mean: "))
        params['alpha'] = float(input("Enter the significance level (default: 0.05): ") or 0.05)
        params['tail'] = input("Enter the type of test ('two', 'left', 'right', default: 'two'): ") or 'two'

        data_input = input(
            "Enter the sample data separated by spaces or press Enter to provide sample mean, size, and std: ")
        if data_input:
            params['data'] = list(map(float, data_input.split()))
        else:
            params['sample_mean'] = float(input("Enter the sample mean: "))
            params['sample_size'] = int(input("Enter the sample size: "))
            params['sample_std'] = float(input("Enter the sample standard deviation: "))

    elif test_type == 'test_proportion':
        params['sample_proportion'] = float(input("Enter the sample proportion: "))
        params['sample_size'] = int(input("Enter the sample size: "))
        params['pop_proportion'] = float(input("Enter the population proportion: "))
        params['alpha'] = float(input("Enter the significance level (default: 0.05): ") or 0.05)
        params['tail'] = input("Enter the type of test ('two', 'left', 'right', default: 'two'): ") or 'two'

    elif test_type == 'test_variance':
        params['pop_variance'] = float(input("Enter the population variance: "))
        params['alpha'] = float(input("Enter the significance level (default: 0.05): ") or 0.05)
        params['tail'] = input("Enter the type of test ('two', 'left', 'right', default: 'two'): ") or 'two'

        data_input = input(
            "Enter the sample data separated by spaces or press Enter to provide sample variance and size: ")
        if data_input:
            params['data'] = list(map(float, data_input.split()))
        else:
            params['sample_variance'] = float(input("Enter the sample variance: "))
            params['sample_size'] = int(input("Enter the sample size: "))

    elif test_type == 'z_test_diff_means':
        params['sample1_mean'] = float(input("Enter the sample mean of group 1: "))
        params['sample1_size'] = int(input("Enter the sample size of group 1: "))
        params['sample2_mean'] = float(input("Enter the sample mean of group 2: "))
        params['sample2_size'] = int(input("Enter the sample size of group 2: "))
        params['pop_std1'] = float(input("Enter the population standard deviation of group 1: "))
        params['pop_std2'] = float(input("Enter the population standard deviation of group 2: "))
        params['delta0'] = float(input("Enter the hypothesized difference between means (default: 0): ") or 0.0)
        params['alpha'] = float(input("Enter the significance level (default: 0.05): ") or 0.05)
        params['tail'] = input("Enter the type of test ('two', 'left', 'right', default: 'two'): ") or 'two'

    elif test_type == 't_test_diff_means':
        params['alpha'] = float(input("Enter the significance level (default: 0.05): ") or 0.05)
        params['tail'] = input("Enter the type of test ('two', 'left', 'right', default: 'two'): ") or 'two'

        data_input1 = input("Enter the sample data for group 1 separated by spaces: ")
        data_input2 = input("Enter the sample data for group 2 separated by spaces: ")

        params['data1'] = list(map(float, data_input1.split()))
        params['data2'] = list(map(float, data_input2.split()))

    elif test_type == 'paired_t_test':
        params['alpha'] = float(input("Enter the significance level (default: 0.05): ") or 0.05)
        params['tail'] = input("Enter the type of test ('two', 'left', 'right', default: 'two'): ") or 'two'

        data_input1 = input("Enter the sample data for group 1 separated by spaces: ")
        data_input2 = input("Enter the sample data for group 2 separated by spaces: ")

        params['data1'] = list(map(float, data_input1.split()))
        params['data2'] = list(map(float, data_input2.split()))

    elif test_type == 'z_test_diff_proportions':
        params['sample1_successes'] = int(input("Enter the number of successes in sample 1: "))
        params['sample1_size'] = int(input("Enter the sample size of group 1: "))
        params['sample2_successes'] = int(input("Enter the number of successes in sample 2: "))
        params['sample2_size'] = int(input("Enter the sample size of group 2: "))
        params['alpha'] = float(input("Enter the significance level (default: 0.05): ") or 0.05)
        params['tail'] = input("Enter the type of test ('two', 'left', 'right', default: 'two'): ") or 'two'

    return params


def main():
    test_types = [
        'z_test_mean', 't_test_mean', 'test_proportion', 'test_variance',
        'z_test_diff_means', 't_test_diff_means', 'paired_t_test', 'z_test_diff_proportions'
    ]

    while True:
        print("Select the test you want to perform:")
        for i, test in enumerate(test_types, 1):
            print(f"{i}. {test.replace('_', ' ').title()}")

        try:
            test_choice = int(input("Enter the number corresponding to the test: ")) - 1
            if 0 <= test_choice < len(test_types):
                test_type = test_types[test_choice]
                break
            else:
                print("Invalid number, please try again.")
        except ValueError:
            print("Invalid input, please enter a number.")

    params = prompt_for_test_parameters(test_type)

    if test_type == 'z_test_mean':
        result = hypothesis_testing_mean_known_std(**params)
    elif test_type == 't_test_mean':
        result = hypothesis_testing_mean_unknown_std(**params)
    elif test_type == 'test_proportion':
        result = hypothesis_testing_proportion(**params)
    elif test_type == 'test_variance':
        result = hypothesis_testing_variance(**params)
    elif test_type == 'z_test_diff_means':
        result = hypothesis_testing_diff_means_known_std(**params)
    elif test_type == 't_test_diff_means':
        result = hypothesis_testing_diff_means_unknown_std(**params)
    elif test_type == 'paired_t_test':
        result = hypothesis_testing_paired_samples(**params)
    elif test_type == 'z_test_diff_proportions':
        result = hypothesis_testing_diff_proportions(**params)

    if result is not None:
        for key, value in result.items():
            print(f"{key}: {value}")

    input("Press Enter to exit...")


# if __name__ == "__main__":
#     main()
