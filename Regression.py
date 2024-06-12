import numpy as np
from scipy.stats import t
import statsmodels.api as sm


def linear_regression_analysis(data1, data2, x_pred=None, alpha=0.05):
    x = np.array(data1)
    y = np.array(data2)

    # Perform linear regression
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    b1 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
    b0 = y_mean - b1 * x_mean

    # Calculate predicted values
    y_pred = b0 + b1 * x

    # Calculate deviations
    total_variation = np.sum((y - y_mean) ** 2)
    explained_deviation = np.sum((y_pred - y_mean) ** 2)
    unexplained_deviation = np.sum((y - y_pred) ** 2)

    # Calculate correlation and determination coefficients
    correlation_coefficient = np.corrcoef(x, y)[0, 1]
    determination_coefficient = correlation_coefficient ** 2

    # Calculate standard error of estimate
    standard_error_estimate = np.sqrt(unexplained_deviation / (n - 2))

    # Prediction interval for a specific value of x
    if x_pred is not None and type(x_pred) is not str:
        x_pred = np.array(x_pred)
        y_pred_pred = b0 + b1 * x_pred
        se_pred = standard_error_estimate * np.sqrt(
            1 + (1 / n) + ((n * ((x_pred - x_mean) ** 2)) / ((n * np.sum(x ** 2)) - (np.sum(x)) ** 2)))
        t_value = t.ppf(1 - alpha / 2, n - 2)
        margin_of_error = t_value * se_pred

        pred_interval_lower = y_pred_pred - margin_of_error
        pred_interval_upper = y_pred_pred + margin_of_error

        y_pred_lower = (b0 - margin_of_error) + b1 * x
        y_pred_upper = (b0 + margin_of_error) + b1 * x
    else:
        pred_interval_lower = None
        pred_interval_upper = None
        y_pred_lower = None
        y_pred_upper = None

    return {
        'b0': b0,
        'b1': b1,
        'correlation_coefficient': correlation_coefficient,
        'determination_coefficient': determination_coefficient,
        'total_deviation': total_variation,
        'explained_deviation': explained_deviation,
        'unexplained_deviation': unexplained_deviation,
        'y_pred': y_pred,
        'x': x,
        'standard_error_estimate': standard_error_estimate,
        'pred_interval_lower': pred_interval_lower,
        'pred_interval_upper': pred_interval_upper,
        'y_pred_lower': y_pred_lower,
        'y_pred_upper': y_pred_upper
    }


def multiple_linear_regression_analysis(X_data, y_data, alpha=0.05):
    X = np.array(X_data)
    y = np.array(y_data)

    # Add a constant (intercept) to the model
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Get predictions
    y_pred = model.predict(X)

    # Get the summary of the model
    model_summary = model.summary()

    # Get standard error of estimate
    residuals = y - y_pred
    standard_error_estimate = np.sqrt(np.sum(residuals**2) / (len(y) - len(model.params)))

    # Get prediction intervals for the fitted values
    pred_interval = model.get_prediction(X).summary_frame(alpha=alpha)
    pred_interval_lower = pred_interval['obs_ci_lower']
    pred_interval_upper = pred_interval['obs_ci_upper']

    return {
        'model': model,
        'y_pred': y_pred,
        'pred_interval_lower': pred_interval_lower,
        'pred_interval_upper': pred_interval_upper,
        'standard_error_estimate': standard_error_estimate,
        'model_summary': model_summary
    }
