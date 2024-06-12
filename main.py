import sys
import traceback

import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox,
                             QPushButton, QGridLayout, QHBoxLayout, QTextEdit, QMessageBox, QFormLayout, QSizePolicy)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.stats import norm, chi2, t, f

from analysis_and_comparison_of_variances import f_test_two_sample_variance, f_test_two_sample_variance_from_stats, \
    one_way_anova
from hypothesis_testing_one_sample import (
    hypothesis_testing_variance,
    hypothesis_testing_proportion,
    hypothesis_testing_mean_known_std,
    hypothesis_testing_mean_unknown_std,
)
from hypothesis_test_two_sample import (
    hypothesis_testing_paired_samples,
    hypothesis_testing_diff_proportions,
    hypothesis_testing_diff_means_known_std,
    hypothesis_testing_diff_means_unknown_std,
)
from Regression import linear_regression_analysis, multiple_linear_regression_analysis

from chi_square_tests_F_distribution import goodness_of_fit_test, chi_square_test_independence


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class HypothesisTestingApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hypothesis Testing Application")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.test_types = [
            'z_test_mean', 't_test_mean', 'test_proportion', 'test_variance',
            'z_test_diff_means', 't_test_diff_means', 'paired_t_test', 'z_test_diff_proportions',
            'linear_regression', 'multiple_regression', 'goodness_of_fit', 'chi_square_independence',
            'f_test_two_sample', 'anova'
        ]

        self.input_fields = {}

        self.create_widgets()

    def create_widgets(self):
        # Test selection dropdown
        self.test_label = QLabel("Select Test:")
        self.test_combo = QComboBox()
        self.test_combo.addItems(self.test_types)
        self.test_combo.currentIndexChanged.connect(self.display_test_form)

        test_selection_layout = QHBoxLayout()
        test_selection_layout.addWidget(self.test_label)
        test_selection_layout.addWidget(self.test_combo)

        self.layout.addLayout(test_selection_layout)

        # Frame for dynamic form
        self.form_layout = QFormLayout()
        self.layout.addLayout(self.form_layout)

        # Submit button
        self.submit_button = QPushButton("Run Test")
        self.submit_button.clicked.connect(self.run_test)
        self.layout.addWidget(self.submit_button)

        # Layout for results and plot
        self.results_plot_layout = QHBoxLayout()
        self.layout.addLayout(self.results_plot_layout)

        # Text area for results
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFixedHeight(150)
        self.result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_plot_layout.addWidget(self.result_text, 1)

        # Canvas for plotting
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_plot_layout.addWidget(self.canvas, 3)

        # Display form for the initial selection
        self.display_test_form()

    def display_test_form(self):
        # Clear previous form
        for i in reversed(range(self.form_layout.count())):
            widget_to_remove = self.form_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                self.form_layout.removeWidget(widget_to_remove)
                widget_to_remove.setParent(None)

        # Clear the input fields dictionary
        self.input_fields.clear()

        # Get selected test type
        test_type = self.test_combo.currentText()

        # Display appropriate form based on test type
        if test_type in ['z_test_mean', 't_test_mean', 'test_variance']:
            self.create_common_form()
            if test_type == 'z_test_mean':
                self.add_input_field("sample_mean", "Sample Mean")
                self.add_input_field("sample_size", "Sample Size")
                self.add_input_field("pop_mean", "Population Mean")
                self.add_input_field("pop_std", "Population Standard Deviation")
            elif test_type == 't_test_mean':
                self.add_input_field("sample_mean", "Sample Mean")
                self.add_input_field("sample_size", "Sample Size")
                self.add_input_field("sample_std", "Sample Standard Deviation")
                self.add_input_field("pop_mean", "Population Mean")
            elif test_type == 'test_variance':
                self.add_input_field("sample_variance", "Sample Variance")
                self.add_input_field("sample_size", "Sample Size")
                self.add_input_field("pop_variance", "Population Variance")
        elif test_type == 'test_proportion':
            self.create_common_form()
            self.add_input_field("sample_proportion", "Sample Proportion")
            self.add_input_field("sample_size", "Sample Size")
            self.add_input_field("pop_proportion", "Population Proportion")
        elif test_type in ['z_test_diff_means', 't_test_diff_means']:
            self.create_diff_means_form()
            if test_type == 'z_test_diff_means':
                self.add_input_field("pop_std1", "Population Std Dev Group 1")
                self.add_input_field("pop_std2", "Population Std Dev Group 2")
            elif test_type == 't_test_diff_means':
                self.add_input_field("sample1_std", "Sample Std Dev Group 1")
                self.add_input_field("sample2_std", "Sample Std Dev Group 2")
                self.add_data_input_fields()
        elif test_type == 'paired_t_test':
            self.create_paired_test_form()
        elif test_type == 'z_test_diff_proportions':
            self.create_diff_proportions_form()
        elif test_type == 'linear_regression':
            self.create_linear_regression_form()
        elif test_type == 'multiple_regression':
            self.create_multiple_regression_form()
        elif test_type == 'goodness_of_fit':
            self.create_goodness_of_fit_form()
        elif test_type == 'chi_square_independence':
            self.create_chi_square_independence_form()
        elif test_type == 'f_test_two_sample':
            self.create_f_test_two_sample_form()
        elif test_type == 'anova':
            self.create_anova_form()

    def create_common_form(self):
        self.add_input_field("alpha", "Significance Level", default="0.05")
        self.add_input_field("tail", "Tail Type (two, left, right)", default="two")

        data_label = QLabel("Sample Data (space-separated)")
        self.data_entry = QLineEdit()
        self.data_entry.setObjectName("data_entry")
        self.form_layout.addRow(data_label, self.data_entry)
        self.input_fields["data"] = self.data_entry

    def create_diff_means_form(self):
        self.create_common_form()
        self.add_input_field("sample1_mean", "Sample Mean Group 1")
        self.add_input_field("sample1_size", "Sample Size Group 1")
        self.add_input_field("sample2_mean", "Sample Mean Group 2")
        self.add_input_field("sample2_size", "Sample Size Group 2")
        self.add_input_field("delta0", "Hypothesized Difference (default 0)", default="0.0")

    def create_paired_test_form(self):
        self.add_input_field("alpha", "Significance Level", default="0.05")
        self.add_input_field("tail", "Tail Type (two, left, right)", default="two")

        data1_label = QLabel("Sample Data Group 1 (space-separated)")
        self.data1_entry = QLineEdit()
        self.data1_entry.setObjectName("data1_entry")
        self.form_layout.addRow(data1_label, self.data1_entry)
        self.input_fields["data1"] = self.data1_entry
        # Debug: confirm widget is added
        print(f"Added data1 input field")

        data2_label = QLabel("Sample Data Group 2 (space-separated)")
        self.data2_entry = QLineEdit()
        self.data2_entry.setObjectName("data2_entry")
        self.form_layout.addRow(data2_label, self.data2_entry)
        self.input_fields["data2"] = self.data2_entry
        # Debug: confirm widget is added
        print(f"Added data2 input field")

    def add_data_input_fields(self):
        data1_label = QLabel("Sample Data Group 1 (space-separated)")
        self.data1_entry = QLineEdit()
        self.data1_entry.setObjectName("data1_entry")
        self.form_layout.addRow(data1_label, self.data1_entry)
        self.input_fields["data1"] = self.data1_entry

        data2_label = QLabel("Sample Data Group 2 (space-separated)")
        self.data2_entry = QLineEdit()
        self.data2_entry.setObjectName("data2_entry")
        self.form_layout.addRow(data2_label, self.data2_entry)
        self.input_fields["data2"] = self.data2_entry

    def create_diff_proportions_form(self):
        self.add_input_field("sample1_successes", "Number of Successes Group 1")
        self.add_input_field("sample1_size", "Sample Size Group 1")
        self.add_input_field("sample2_successes", "Number of Successes Group 2")
        self.add_input_field("sample2_size", "Sample Size Group 2")
        self.add_input_field("alpha", "Significance Level", default="0.05")
        self.add_input_field("tail", "Tail Type (two, left, right)", default="two")

    def create_linear_regression_form(self):
        data1_label = QLabel("X Data (space-separated)")
        self.data1_entry = QLineEdit()
        self.data1_entry.setObjectName("data1_entry")
        self.form_layout.addRow(data1_label, self.data1_entry)
        self.input_fields["data1"] = self.data1_entry

        data2_label = QLabel("Y Data (space-separated)")
        self.data2_entry = QLineEdit()
        self.data2_entry.setObjectName("data2_entry")
        self.form_layout.addRow(data2_label, self.data2_entry)
        self.input_fields["data2"] = self.data2_entry

        x_pred_label = QLabel("X Values for Prediction Interval (space-separated)")
        self.x_pred_entry = QLineEdit()
        self.x_pred_entry.setObjectName("x_pred_entry")
        self.form_layout.addRow(x_pred_label, self.x_pred_entry)
        self.input_fields["x_pred"] = self.x_pred_entry

        self.add_input_field("alpha", "Significance Level", default="0.05")

    def create_multiple_regression_form(self):
        dataX_label = QLabel("X Data (comma-separated, space-separated values per row)")
        self.dataX_entry = QTextEdit()
        self.dataX_entry.setObjectName("dataX_entry")
        self.form_layout.addRow(dataX_label, self.dataX_entry)
        self.input_fields["dataX"] = self.dataX_entry

        dataY_label = QLabel("Y Data (space-separated)")
        self.dataY_entry = QLineEdit()
        self.dataY_entry.setObjectName("dataY_entry")
        self.form_layout.addRow(dataY_label, self.dataY_entry)
        self.input_fields["dataY"] = self.dataY_entry

    def create_goodness_of_fit_form(self):
        observed_label = QLabel("Observed Frequencies (space-separated)")
        self.observed_entry = QLineEdit()
        self.observed_entry.setObjectName("observed_entry")
        self.form_layout.addRow(observed_label, self.observed_entry)
        self.input_fields["observed"] = self.observed_entry

        expected_label = QLabel("Expected Frequencies (space-separated)")
        self.expected_entry = QLineEdit()
        self.expected_entry.setObjectName("expected_entry")
        self.form_layout.addRow(expected_label, self.expected_entry)
        self.input_fields["expected"] = self.expected_entry

        alpha_label = QLabel("Significance Level")
        self.alpha_entry = QLineEdit()
        self.alpha_entry.setObjectName("alpha_entry")
        self.alpha_entry.setText("0.05")
        self.form_layout.addRow(alpha_label, self.alpha_entry)
        self.input_fields["alpha"] = self.alpha_entry

    def create_chi_square_independence_form(self):
        observed_label = QLabel("Observed Frequencies (comma-separated, space-separated values per row)")
        self.observed_entry = QTextEdit()
        self.observed_entry.setObjectName("observed_entry")
        self.form_layout.addRow(observed_label, self.observed_entry)
        self.input_fields["observed"] = self.observed_entry

        alpha_label = QLabel("Significance Level")
        self.alpha_entry = QLineEdit()
        self.alpha_entry.setObjectName("alpha_entry")
        self.alpha_entry.setText("0.05")
        self.form_layout.addRow(alpha_label, self.alpha_entry)
        self.input_fields["alpha"] = self.alpha_entry

    def create_f_test_two_sample_form(self):
        self.create_common_form()

        data1_label = QLabel("Sample Data Group 1 (space-separated)")
        self.data1_entry = QLineEdit()
        self.data1_entry.setObjectName("data1_entry")
        self.form_layout.addRow(data1_label, self.data1_entry)
        self.input_fields["data1"] = self.data1_entry

        data2_label = QLabel("Sample Data Group 2 (space-separated)")
        self.data2_entry = QLineEdit()
        self.data2_entry.setObjectName("data2_entry")
        self.form_layout.addRow(data2_label, self.data2_entry)
        self.input_fields["data2"] = self.data2_entry

        sample1_variance_label = QLabel("Sample Variance Group 1 (if no data)")
        self.sample1_variance_entry = QLineEdit()
        self.sample1_variance_entry.setObjectName("sample1_variance_entry")
        self.form_layout.addRow(sample1_variance_label, self.sample1_variance_entry)
        self.input_fields["sample1_variance"] = self.sample1_variance_entry

        sample1_size_label = QLabel("Sample Size Group 1 (if no data)")
        self.sample1_size_entry = QLineEdit()
        self.sample1_size_entry.setObjectName("sample1_size_entry")
        self.form_layout.addRow(sample1_size_label, self.sample1_size_entry)
        self.input_fields["sample1_size"] = self.sample1_size_entry

        sample2_variance_label = QLabel("Sample Variance Group 2 (if no data)")
        self.sample2_variance_entry = QLineEdit()
        self.sample2_variance_entry.setObjectName("sample2_variance_entry")
        self.form_layout.addRow(sample2_variance_label, self.sample2_variance_entry)
        self.input_fields["sample2_variance"] = self.sample2_variance_entry

        sample2_size_label = QLabel("Sample Size Group 2 (if no data)")
        self.sample2_size_entry = QLineEdit()
        self.sample2_size_entry.setObjectName("sample2_size_entry")
        self.form_layout.addRow(sample2_size_label, self.sample2_size_entry)
        self.input_fields["sample2_size"] = self.sample2_size_entry

    def create_anova_form(self):
        self.create_common_form()

        for i in range(3):  # Initial three groups
            data_label = QLabel(f"Sample Data Group {i + 1} (space-separated)")
            data_entry = QLineEdit()
            data_entry.setObjectName(f"data_group_{i + 1}")
            self.form_layout.addRow(data_label, data_entry)
            self.input_fields[f"data_group_{i + 1}"] = data_entry

        add_group_button = QPushButton("Add Group")
        add_group_button.clicked.connect(self.add_anova_group)
        self.form_layout.addRow(add_group_button)
        self.add_group_button = add_group_button
        self.group_counter = 3

    def add_anova_group(self):
        self.group_counter += 1
        data_label = QLabel(f"Sample Data Group {self.group_counter} (space-separated)")
        data_entry = QLineEdit()
        data_entry.setObjectName(f"data_group_{self.group_counter}")
        self.form_layout.addRow(data_label, data_entry)
        self.input_fields[f"data_group_{self.group_counter}"] = data_entry

    def add_input_field(self, var_name, label_text, default=""):
        label = QLabel(label_text)
        entry = QLineEdit()
        entry.setObjectName(var_name)
        entry.setText(default)
        self.form_layout.addRow(label, entry)
        self.input_fields[var_name] = entry
        # Debug: confirm widget is added
        print(f"Added {var_name} input field")

    def run_test(self):
        test_type = self.test_combo.currentText()
        params = {}

        # Debug: Print all widgets in input_fields
        # for key, widget in self.input_fields.items():
        #     print(f"Widget name: {key}, value: {widget.text()}")

        for var_name, widget in self.input_fields.items():
            if ((test_type in ['test_proportion'] or test_type in ['z_test_diff_means']
                 or test_type in ['z_test_diff_proportions'] or test_type in ['t_test_diff_means']
                 or test_type in ['f_test_two_sample'])
                    and var_name == "data"):
                continue
            if var_name == 'observed':
                params[var_name] = widget.toPlainText()
            if var_name and (test_type != 'chi_square_independence'):
                try:
                    params[var_name] = float(widget.text()) if '.' in widget.text() else int(widget.text())
                except ValueError:
                    params[var_name] = widget.text()

        if ("data" in self.input_fields and (test_type not in ['test_proportion']
                                             or test_type not in ['z_test_diff_means'] or test_type not in [
                                                 'z_test_diff_proportions']
                                             or test_type not in ['t_test_diff_means']
                                             or test_type not in ['f_test_two_sample'])):
            data_input = self.input_fields["data"].text()
            if data_input:
                params['data'] = list(map(float, data_input.split()))
        if ("data1" in self.input_fields and "data2" in self.input_fields):
            data_input1 = self.input_fields["data1"].text()
            data_input2 = self.input_fields["data2"].text()
            if data_input1 and data_input2:
                params['data1'] = list(map(float, data_input1.split()))
                params['data2'] = list(map(float, data_input2.split()))
            else:
                params['data1'] = None
                params['data2'] = None
        if test_type == 'multiple_regression':
            params['dataX'] = self.input_fields['dataX'].toPlainText()
            params['dataY'] = self.input_fields['dataY'].text()

        try:
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
            elif test_type == 'linear_regression':
                result = linear_regression_analysis(**params)
            elif test_type == 'multiple_regression':
                dataX = [list(map(float, row.split())) for row in params['dataX'].strip().split('\n')]
                dataY = list(map(float, params['dataY'].strip().split()))
                result = multiple_linear_regression_analysis(dataX, dataY)
            elif test_type == 'goodness_of_fit':
                observed = list(map(float, params['observed'].split()))
                expected = list(map(float, params['expected'].split()))
                alpha = float(params.get('alpha', 0.05))
                result = goodness_of_fit_test(observed, expected, alpha)
            elif test_type == 'chi_square_independence':
                observed = np.array([list(map(float, row.split())) for row in params['observed'].strip().split('\n')])
                alpha = float(params.get('alpha', 0.05))
                result = chi_square_test_independence(observed, alpha)
            elif test_type == 'f_test_two_sample':
                data1 = list(map(float, self.input_fields["data1"].text().split())) if self.input_fields[
                    "data1"].text() else None
                data2 = list(map(float, self.input_fields["data2"].text().split())) if self.input_fields[
                    "data2"].text() else None
                alpha = float(params.get('alpha', 0.05))
                tail = params.get('tail', 'two')

                if data1 is not None and data2 is not None:
                    result = f_test_two_sample_variance(data1, data2, alpha, tail)
                else:
                    sample1_variance = float(params.get('sample1_variance'))
                    sample1_size = int(params.get('sample1_size'))
                    sample2_variance = float(params.get('sample2_variance'))
                    sample2_size = int(params.get('sample2_size'))
                    result = f_test_two_sample_variance_from_stats(sample1_variance, sample1_size,
                                                                   sample2_variance, sample2_size, alpha,
                                                                   tail)
            elif test_type == 'anova':
                groups = []
                for i in range(1, self.group_counter + 1):
                    data_group = self.input_fields.get(f"data_group_{i}")
                    if data_group:
                        group_data = list(map(float, data_group.text().split()))
                        if group_data:
                            groups.append(group_data)
                if groups:
                    alpha = float(params.get('alpha', 0.05))
                    result = one_way_anova(alpha, *groups)

            self.result_text.clear()

            for key, value in result.items():
                self.result_text.append(f"{key}: {value}")

            self.plot_result(result, test_type, params)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_result(self, result, test_type, params):
        self.canvas.axes.cla()

        # Plots for different test types
        if test_type in ['z_test_mean', 'test_proportion', 'z_test_diff_means', 'z_test_diff_proportions']:
            z_statistic = result['z']
            z_critical = result['z_critical']
            alpha = result['alpha']
            tail = result['tail']

            # Plot standard normal distribution
            x = np.linspace(-4, 4, 1000)
            y = norm.pdf(x)
            self.canvas.axes.plot(x, y, label='Standard Normal Distribution')

            # Determine critical values and rejection regions
            if tail == 'two':
                self.canvas.axes.fill_between(x, 0, y, where=(x <= -z_critical) | (x >= z_critical), color='red',
                                              alpha=0.5, label='Rejection Region')
                self.canvas.axes.axvline(-z_critical, color='red', linestyle='--',
                                         label=f'Critical Value = {-z_critical:.2f}')
                self.canvas.axes.axvline(z_critical, color='red', linestyle='--',
                                         label=f'Critical Value = {z_critical:.2f}')
            elif tail == 'left':
                self.canvas.axes.fill_between(x, 0, y, where=(x <= z_critical), color='red', alpha=0.5,
                                              label='Rejection Region')
                self.canvas.axes.axvline(z_critical, color='red', linestyle='--',
                                         label=f'Critical Value = {z_critical:.2f}')
            elif tail == 'right':
                self.canvas.axes.fill_between(x, 0, y, where=(x >= z_critical), color='red', alpha=0.5,
                                              label='Rejection Region')
                self.canvas.axes.axvline(z_critical, color='red', linestyle='--',
                                         label=f'Critical Value = {z_critical:.2f}')

            # Plot the test statistic
            self.canvas.axes.axvline(z_statistic, color='blue', linestyle='-', linewidth=2,
                                     label=f'Test Statistic = {z_statistic:.2f}')

            # Add titles and labels
            if test_type == 'z_test_mean':
                title = f"Z-Test for Mean with Known Std Dev ({tail.capitalize()}-Tailed Test)"
            elif test_type == 'test_proportion':
                title = f"Z-Test for Proportion ({tail.capitalize()}-Tailed Test)"
            else:  # z_test_diff_means
                title = f"Z-Test for Difference Between Means ({tail.capitalize()}-Tailed Test)"

            self.canvas.axes.set_title(title)
            self.canvas.axes.set_xlabel('Z-Value')
            self.canvas.axes.set_ylabel('Probability Density')
            self.canvas.axes.legend(loc='upper left')

        elif test_type in ['t_test_mean', 't_test_diff_means']:
            t_statistic = result['t']
            t_critical = result['t_critical']
            alpha = result['alpha']
            tail = result['tail']
            df = result['degrees_of_freedom']

            # Plot t distribution
            x = np.linspace(-4, 4, 1000)
            y = t.pdf(x, df)
            self.canvas.axes.plot(x, y, label=f'T Distribution (df={df})')

            # Determine critical values and rejection regions
            if tail == 'two':
                self.canvas.axes.fill_between(x, 0, y, where=(x <= -t_critical) | (x >= t_critical), color='red',
                                              alpha=0.5, label='Rejection Region')
                self.canvas.axes.axvline(-t_critical, color='red', linestyle='--',
                                         label=f'Critical Value = {-t_critical:.2f}')
                self.canvas.axes.axvline(t_critical, color='red', linestyle='--',
                                         label=f'Critical Value = {t_critical:.2f}')
            elif tail == 'left':
                self.canvas.axes.fill_between(x, 0, y, where=(x <= t_critical), color='red', alpha=0.5,
                                              label='Rejection Region')
                self.canvas.axes.axvline(t_critical, color='red', linestyle='--',
                                         label=f'Critical Value = {t_critical:.2f}')
            elif tail == 'right':
                self.canvas.axes.fill_between(x, 0, y, where=(x >= t_critical), color='red', alpha=0.5,
                                              label='Rejection Region')
                self.canvas.axes.axvline(t_critical, color='red', linestyle='--',
                                         label=f'Critical Value = {t_critical:.2f}')

            # Plot the test statistic
            self.canvas.axes.axvline(t_statistic, color='blue', linestyle='-', linewidth=2,
                                     label=f'Test Statistic = {t_statistic:.2f}')

            # Add titles and labels
            if test_type == 't_test_mean':
                title = f"T-Test for Mean with Unknown Std Dev ({tail.capitalize()}-Tailed Test)"
            else:  # t_test_diff_means
                title = f"T-Test for Difference Between Means ({tail.capitalize()}-Tailed Test)"

            self.canvas.axes.set_title(title)
            self.canvas.axes.set_xlabel('T-Value')
            self.canvas.axes.set_ylabel('Probability Density')
            self.canvas.axes.legend(loc='upper left')

        elif test_type in ['test_variance', 'paired_t_test']:
            if test_type == 'test_variance':
                chi2_statistic = result['chi2_statistic']
                chi2_critical_low = result.get('chi2_critical_low', None)
                chi2_critical_high = result['chi2_critical_high']
                df = result['sample_size'] - 1

                # Plot chi-square distribution
                x = np.linspace(0, chi2.ppf(0.99, df), 1000)
                y = chi2.pdf(x, df)
                self.canvas.axes.plot(x, y, label=f'Chi-Square Distribution (df={df})')

                # Determine critical values and rejection regions
                if chi2_critical_low is not None and result['tail'] == 'two':
                    self.canvas.axes.fill_between(x, 0, y, where=(x <= chi2_critical_low) | (x >= chi2_critical_high),
                                                  color='red', alpha=0.5, label='Rejection Region')
                    self.canvas.axes.axvline(chi2_critical_low, color='red', linestyle='--',
                                             label=f'Critical Value = {chi2_critical_low:.2f}')
                    self.canvas.axes.axvline(chi2_critical_high, color='red', linestyle='--',
                                             label=f'Critical Value = {chi2_critical_high:.2f}')
                elif result['tail'] == 'left':
                    self.canvas.axes.fill_between(x, 0, y, where=(x <= chi2_critical_high), color='red', alpha=0.5,
                                                  label='Rejection Region')
                    self.canvas.axes.axvline(chi2_critical_high, color='red', linestyle='--',
                                             label=f'Critical Value = {chi2_critical_high:.2f}')
                elif result['tail'] == 'right':
                    self.canvas.axes.fill_between(x, 0, y, where=(x >= chi2_critical_high), color='red', alpha=0.5,
                                                  label='Rejection Region')
                    self.canvas.axes.axvline(chi2_critical_high, color='red', linestyle='--',
                                             label=f'Critical Value = {chi2_critical_high:.2f}')

                # Plot the test statistic
                self.canvas.axes.axvline(chi2_statistic, color='blue', linestyle='-', linewidth=2,
                                         label=f'Test Statistic = {chi2_statistic:.2f}')

                # Add titles and labels
                self.canvas.axes.set_title(f"Chi-Square Test for Variance")
                self.canvas.axes.set_xlabel('Chi-Square Value')
                self.canvas.axes.set_ylabel('Probability Density')
                self.canvas.axes.legend(loc='upper right')

            elif test_type == 'paired_t_test':
                t_statistic = result['t']
                t_critical = result['t_critical']
                alpha = result['alpha']
                tail = result['tail']
                df = result['degrees_of_freedom']

                # Plot t distribution
                x = np.linspace(-4, 4, 1000)
                y = t.pdf(x, df)
                self.canvas.axes.plot(x, y, label=f'T Distribution (df={df})')

                # Determine critical values and rejection regions
                if tail == 'two':
                    self.canvas.axes.fill_between(x, 0, y, where=(x <= -t_critical) | (x >= t_critical), color='red',
                                                  alpha=0.5, label='Rejection Region')
                    self.canvas.axes.axvline(-t_critical, color='red', linestyle='--',
                                             label=f'Critical Value = {-t_critical:.2f}')
                    self.canvas.axes.axvline(t_critical, color='red', linestyle='--',
                                             label=f'Critical Value = {t_critical:.2f}')
                elif tail == 'left':
                    self.canvas.axes.fill_between(x, 0, y, where=(x <= t_critical), color='red', alpha=0.5,
                                                  label='Rejection Region')
                    self.canvas.axes.axvline(t_critical, color='red', linestyle='--',
                                             label=f'Critical Value = {t_critical:.2f}')
                elif tail == 'right':
                    self.canvas.axes.fill_between(x, 0, y, where=(x >= t_critical), color='red', alpha=0.5,
                                                  label='Rejection Region')
                    self.canvas.axes.axvline(t_critical, color='red', linestyle='--',
                                             label=f'Critical Value = {t_critical:.2f}')

                # Plot the test statistic
                self.canvas.axes.axvline(t_statistic, color='blue', linestyle='-', linewidth=2,
                                         label=f'Test Statistic = {t_statistic:.2f}')

                # Add titles and labels
                self.canvas.axes.set_title(f"Paired T-Test for Mean Differences ({tail.capitalize()}-Tailed Test)")
                self.canvas.axes.set_xlabel('T-Value')
                self.canvas.axes.set_ylabel('Probability Density')
                self.canvas.axes.legend(loc='upper left')
        elif test_type == 'linear_regression':
            b0 = result['b0']
            b1 = result['b1']
            y_pred = result['y_pred']
            x = np.array(params['data1'])
            y = np.array(params['data2'])
            x_pred = np.array(params['x_pred']) if 'x_pred' in params else None
            pred_interval_lower = result['pred_interval_lower']
            pred_interval_upper = result['pred_interval_upper']
            y_pred_lower = result['y_pred_lower']
            y_pred_upper = result['y_pred_upper']

            # Plot the regression line
            self.canvas.axes.scatter(x, y, color='blue', label='Data Points')
            self.canvas.axes.plot(x, y_pred, color='red', label=f'Regression Line: y = {b0:.2f} + {b1:.2f}x')

            if x_pred is not None and x_pred != "":
                x = np.sort(x)
                y_pred_upper = np.sort(y_pred_upper)
                y_pred_lower = np.sort(y_pred_lower)
                self.canvas.axes.fill_between(x, y_pred_upper, y_pred_lower, color='gray', alpha=0.2,
                                              label='Prediction Interval')
                self.canvas.axes.plot(x, y_pred_lower, color='green', linestyle='--')
                self.canvas.axes.plot(x, y_pred_upper, color='green', linestyle='--')
            else:
                x_pred = None

            # Add titles and labels
            self.canvas.axes.set_title("Linear Regression")
            self.canvas.axes.set_xlabel('X Data')
            self.canvas.axes.set_ylabel('Y Data')
            self.canvas.axes.legend(loc='upper left')

            # Display deviations
            total_deviation = result['total_deviation']
            explained_deviation = result['explained_deviation']
            unexplained_deviation = result['unexplained_deviation']
            correlation_coefficient = result['correlation_coefficient']
            determination_coefficient = result['determination_coefficient']
            standard_error_estimate = result['standard_error_estimate']

            self.result_text.append(f"Total Deviation: {total_deviation:.2f}")
            self.result_text.append(f"Explained Deviation: {explained_deviation:.2f}")
            self.result_text.append(f"Unexplained Deviation: {unexplained_deviation:.2f}")
            self.result_text.append(f"Correlation Coefficient: {correlation_coefficient:.2f}")
            self.result_text.append(f"Determination Coefficient: {determination_coefficient:.2f}")
            self.result_text.append(f"Standard Error of Estimate: {standard_error_estimate:.2f}")

        if test_type == 'multiple_regression':
            model = result['model']
            y_pred = result['y_pred']
            X = np.array([list(map(float, row.split())) for row in params['dataX'].strip().split('\n')])
            y = np.array(list(map(float, params['dataY'].strip().split())))
            pred_interval_lower = result['pred_interval_lower']
            pred_interval_upper = result['pred_interval_upper']

            # Plot the predicted vs actual values
            self.canvas.axes.scatter(y, y_pred, color='blue', label='Predicted vs Actual')
            self.canvas.axes.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal Line')

            # Add titles and labels
            self.canvas.axes.set_title("Multiple Linear Regression")
            self.canvas.axes.set_xlabel('Actual Y Values')
            self.canvas.axes.set_ylabel('Predicted Y Values')
            self.canvas.axes.legend(loc='upper left')

            # Display model summary in the text box
            self.result_text.append(str(result['model_summary']))

        if test_type == 'goodness_of_fit':
            observed = list(map(float, params['observed'].split()))
            expected = list(map(float, params['observed'].split()))
            chi2_stat = result['chi2_stat']
            p_value = result['p_value']
            critical_value = result['critical_value']
            reject_null = result['reject_null']
            alpha = float(params.get('alpha', 0.05))
            df = len(observed) - 1

            # Generate the chi-square distribution
            x = np.linspace(0, chi2.ppf(0.999, df), 1000)
            y = chi2.pdf(x, df)

            # Plot the chi-square distribution
            self.canvas.axes.plot(x, y, label=f'Chi-Square Distribution (df={df})')

            # Fill the rejection region
            self.canvas.axes.fill_between(x, y, where=(x > critical_value), color='red', alpha=0.5,
                                          label='Rejection Region')

            # Plot the chi-square statistic
            self.canvas.axes.axvline(chi2_stat, color='blue', linestyle='--',
                                     label=f'Chi-Square Statistic: {chi2_stat:.4f}')

            # Add titles and labels
            self.canvas.axes.set_title("Chi-Square Goodness-of-Fit Test")
            self.canvas.axes.set_xlabel('Chi-Square Value')
            self.canvas.axes.set_ylabel('Probability Density')
            self.canvas.axes.legend(loc='upper right')

            # Display test results
            # self.result_text.append(f"Chi-Square Statistic: {chi2_stat:.4f}")
            # self.result_text.append(f"P-Value: {p_value:.4f}")
            # self.result_text.append(f"Critical Value: {critical_value:.4f}")
            # self.result_text.append(f"Reject Null Hypothesis: {reject_null}")

        elif test_type == 'chi_square_independence':
            observed = np.array([list(map(float, row.split())) for row in params['observed'].strip().split('\n')])
            chi2_stat = result['chi2_stat']
            p_value = result['p_value']
            critical_value = result['critical_value']
            reject_null = result['reject_null']
            alpha = float(params.get('alpha', 0.05))
            df = (observed.shape[0] - 1) * (observed.shape[1] - 1)

            # Generate the chi-square distribution
            x = np.linspace(0, chi2.ppf(0.999, df), 1000)
            y = chi2.pdf(x, df)

            # Plot the chi-square distribution
            self.canvas.axes.plot(x, y, label=f'Chi-Square Distribution (df={df})')

            # Fill the rejection region
            self.canvas.axes.fill_between(x, y, where=(x > critical_value), color='red', alpha=0.5,
                                          label='Rejection Region')

            # Plot the chi-square statistic
            self.canvas.axes.axvline(chi2_stat, color='blue', linestyle='--',
                                     label=f'Chi-Square Statistic: {chi2_stat:.4f}')

            # Add titles and labels
            self.canvas.axes.set_title("Chi-Square Test for Independence")
            self.canvas.axes.set_xlabel('Chi-Square Value')
            self.canvas.axes.set_ylabel('Probability Density')
            self.canvas.axes.legend(loc='upper right')

            # Display test results
            # self.result_text.append(f"Chi-Square Statistic: {chi2_stat:.4f}")
            # self.result_text.append(f"P-Value: {p_value:.4f}")
            # self.result_text.append(f"Critical Value: {critical_value:.4f}")
            # self.result_text.append(f"Reject Null Hypothesis: {reject_null}")

        elif test_type == 'f_test_two_sample':
            f_stat = result['f_stat']
            p_value = result['p_value']
            dfn = result['dfn']
            dfd = result['dfd']
            reject_null = result['reject_null']
            critical_values = result['critical_values']
            alpha = float(params.get('alpha', 0.05))

            # Generate the F-distribution
            x = np.linspace(0, f.ppf(0.999, dfn, dfd), 1000)
            y = f.pdf(x, dfn, dfd)

            # Plot the F-distribution
            self.canvas.axes.plot(x, y, label=f'F-Distribution (dfn={dfn}, dfd={dfd})')

            # Fill the rejection region
            if len(critical_values) == 2:
                critical_value_low, critical_value_high = critical_values
                self.canvas.axes.fill_between(x, y, where=(x < critical_value_low) | (x > critical_value_high),
                                              color='red', alpha=0.5, label='Rejection Region')
                self.canvas.axes.axvline(critical_value_low, color='green', linestyle='--',
                                         label=f'Critical Value Low: {critical_value_low:.4f}')
                self.canvas.axes.axvline(critical_value_high, color='green', linestyle='--',
                                         label=f'Critical Value High: {critical_value_high:.4f}')
            else:
                critical_value = critical_values[0]
                self.canvas.axes.fill_between(x, y, where=(x > critical_value), color='red', alpha=0.5,
                                              label='Rejection Region')
                self.canvas.axes.axvline(critical_value, color='green', linestyle='--',
                                         label=f'Critical Value: {critical_value:.4f}')

            # Plot the F-statistic
            self.canvas.axes.axvline(f_stat, color='blue', linestyle='--', label=f'F-Statistic: {f_stat:.4f}')

            # Add titles and labels
            self.canvas.axes.set_title("F-Test for Two Sample Variances")
            self.canvas.axes.set_xlabel('F Value')
            self.canvas.axes.set_ylabel('Probability Density')
            self.canvas.axes.legend(loc='upper right')

            # Display test results
            # self.result_text.append(f"F-Statistic: {f_stat:.4f}")
            # self.result_text.append(f"P-Value: {p_value:.4f}")
            # self.result_text.append(f"Degrees of Freedom (Numerator): {dfn}")
            # self.result_text.append(f"Degrees of Freedom (Denominator): {dfd}")
            # self.result_text.append(f"Critical Values: {', '.join([f'{cv:.4f}' for cv in critical_values])}")
            # self.result_text.append(f"Reject Null Hypothesis: {reject_null}")

        elif test_type == 'anova':
            f_stat = result['f_stat']
            p_value = result['p_value']
            df_between = result['df_between']
            df_within = result['df_within']
            critical_value = result['critical_value']
            reject_null = result['reject_null']
            alpha = float(params.get('alpha', 0.05))

            # Generate the F-distribution
            x = np.linspace(0, f.ppf(0.999, df_between, df_within), 1000)
            y = f.pdf(x, df_between, df_within)

            # Plot the F-distribution
            self.canvas.axes.plot(x, y, label=f'F-Distribution (df_between={df_between}, df_within={df_within})')

            # Fill the rejection region
            self.canvas.axes.fill_between(x, y, where=(x > critical_value), color='red', alpha=0.5,
                                          label='Rejection Region')
            self.canvas.axes.axvline(critical_value, color='green', linestyle='--',
                                     label=f'Critical Value: {critical_value:.4f}')

            # Plot the F-statistic
            self.canvas.axes.axvline(f_stat, color='blue', linestyle='--', label=f'F-Statistic: {f_stat:.4f}')

            # Add titles and labels
            self.canvas.axes.set_title("One-Way ANOVA")
            self.canvas.axes.set_xlabel('F Value')
            self.canvas.axes.set_ylabel('Probability Density')
            self.canvas.axes.legend(loc='upper right')

            # Display test results
            # self.result_text.append(f"F-Statistic: {f_stat:.4f}")
            # self.result_text.append(f"P-Value: {p_value:.4f}")
            # self.result_text.append(f"Degrees of Freedom (Between Groups): {df_between}")
            # self.result_text.append(f"Degrees of Freedom (Within Groups): {df_within}")
            # self.result_text.append(f"Critical Value: {critical_value:.4f}")
            # self.result_text.append(f"Reject Null Hypothesis: {reject_null}")

        self.canvas.draw()


def main():
    try:
        app = QApplication(sys.argv)
        window = HypothesisTestingApp()  # Replace with your main window class
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        error_message = f"An error occurred: {e}\n\n{traceback.format_exc()}"
        QMessageBox.critical(None, "Error", error_message)
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
