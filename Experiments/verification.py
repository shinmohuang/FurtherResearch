import argparse
import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from maraboupy import Marabou
from Scripts.my_deco import running_time, debug, suppress_error
from Scripts import my_func as mf

# Constants
# DEFAULT_FEATURES_NUM = 63
# Constants
DEFAULT_CONFIG_FILE = 'config.ini'

# Function definitions...

def get_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config
# Function definitions (is_within_range, plot_fatigue_distribution)...

def load_data(csv_file_path):
    # Load high fatigue data from CSV and extract values
    data = pd.read_csv(csv_file_path)
    return data['mean'][:-1].values, data['min'][:-1].values, data['max'][:-1].values, data['std'][:-1].values

def solve_model(model_file_path, inputVars, outputVars,
                mean_values, std_values, min_values, max_values, features_num,
                initial_range_factor=0.01, step_size_factor=0.01):
    # Initial range and step size setup
    min_unsat_range = None
    initial_range = [initial_range_factor] * features_num
    step_size = std_values * step_size_factor

    # Marabou options configuration for solving
    options = Marabou.createOptions(snc=True, splittingStrategy='auto', numWorkers=16,
                                    sncSplittingStrategy='auto', restoreTreeStates=True,
                                    splitThreshold=20, solveWithMILP=False, dumpBounds=True)

    # Iterative solving loop
    unsat = True
    ITERATION = 0
    global values
    while unsat:
        print(f"Iteration {ITERATION}")
        ITERATION += 1

        # Re-load the model for each iteration
        network = Marabou.read_onnx(model_file_path)

        # Set the input range for the current iteration
        mf.set_input_range(network, inputVars, mean_values, initial_range)

        # Calculate lower and upper bounds
        lower_bounds = mean_values - initial_range
        upper_bounds = mean_values + initial_range

        # Check if bounds exceed min and max limits
        if np.any(lower_bounds < min_values) or np.any(upper_bounds > max_values):
            print("Bounds exceeded the limits of the dataset.")
            break

        # Define the output conditions for the model
        mf.define_output_conditions(network, outputVars, 2)

        # Solve the model with current configuration
        result = network.solve(verbose=True, options=options)
        status, values, stats = result

        # Check the results and update the range if necessary
        unsat, initial_range = mf.check_results(status, initial_range, step_size)

        if status == "sat":
            print("Solution found!")
            print("Iteration stopped at:", ITERATION)
            return status, initial_range, values
        elif status == "unsat":
            print("No solution found in this iteration.")
            print("Iteration stopped at:", ITERATION)
            print("Time elapsed:", stats.getTotalTimeInMicro())

            # Update for the next iteration
            min_unsat_range = initial_range.copy()

    return status, min_unsat_range, None  # In case of no solution


@running_time
@suppress_error
def main(config_file, model_file_path=None, csv_file_path=None, features_num=None):
    config = get_config(config_file)

    # Use command-line arguments if provided, else use config file
    model_file_path = model_file_path or config['Model']['model_file_path']
    csv_file_path = csv_file_path or config['Dataset']['csv_file_path']
    features_num = int(features_num or config['Dataset']['features_num'])

    # Load the ONNX model using Marabou
    network = Marabou.read_onnx(model_file_path)

    # Identify input and output variables from the network
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]

    # Read data from CSV
    mean_values, min_values, max_values, std_values = load_data(csv_file_path)

    # Solve the model
    status, min_unsat_range, values = solve_model(model_file_path, inputVars, outputVars,
                                 mean_values, std_values, min_values, max_values, features_num)

    # Write the values to CSV
    if values is not None:
        mf.write_values_to_csv(values, 'counterExamples.csv', __file__)
        # min_unsat_range = initial_range.copy()
        range_bounds = [(mean_val - range_val, mean_val + range_val) for mean_val, range_val in
                        zip(mean_values, min_unsat_range)]
        print("最小 UNSAT 范围的界限:", range_bounds)
        # 可以选择将界限写入文件
        mf.write_values_to_csv(range_bounds, 'min_unsat_range.csv', __file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Marabou model verification.')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_FILE, help='Path to the configuration file.')
    parser.add_argument('--model_file_path', type=str, help='Path to the ONNX model file.')
    parser.add_argument('--csv_file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('--features_num', type=int, help='Number of features to consider.')

    args = parser.parse_args()
    main(args.config, args.model_file_path, args.csv_file_path, args.features_num)
