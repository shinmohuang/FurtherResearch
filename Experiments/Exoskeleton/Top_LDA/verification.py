from maraboupy import Marabou
from Scripts.my_deco import running_time, debug, suppress_error
from Scripts import my_func as mf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FEATURESNUM = 10

# Function to check if values are within the high fatigue range
def is_within_range(values, mean_values, std_values):
    # Convert numpy arrays to lists if necessary
    values = values.tolist() if isinstance(values, np.ndarray) else values
    mean_values = mean_values.tolist() if isinstance(mean_values, np.ndarray) else mean_values
    std_values = std_values.tolist() if isinstance(std_values, np.ndarray) else std_values

    lower_bounds = [mean - std for mean, std in zip(mean_values, std_values)]
    upper_bounds = [mean + std for mean, std in zip(mean_values, std_values)]
    return all(lower <= value <= upper for value, lower, upper in zip(values, lower_bounds, upper_bounds))

# Function to plot high fatigue distribution and Marabou output
def plot_fatigue_distribution(high_fatigue, output_values):
    features = high_fatigue['Features'][:10]  # Extract first 10 features
    mean_values = high_fatigue['mean'][:10].values
    std_values = high_fatigue['std'][:10].values

    for i, feature in enumerate(features):
        plt.figure(figsize=(6, 4))  # Set the figure size for each plot
        plt.errorbar(i, mean_values[i], yerr=std_values[i], fmt='o', label='Mean ± STD', color='blue')
        plt.scatter(i, output_values[i], color='red', label='Model Output')

        plt.xticks(range(len(features)), features, rotation='vertical')
        plt.ylabel('Value')
        plt.title(f'Feature: {feature}')

        # Adding legend in the first plot only to avoid repetition
        if i == 0:
            plt.legend()

    plt.show()

# Main function with decorators for running time and error suppression
@running_time
@suppress_error
def main():
    # Define the file path for the ONNX model and the CSV file
    model_file_path = '/home/adam/FurtherResearch/Model/Exoskeleton/Top_LDA/exo_model_top10_without_softmax.onnx'
    csv_file_path = '/home/adam/FurtherResearch/Experiments/Exoskeleton/Top_LDA/dataset_statistics_mapped_fatigue_level_high.csv'

    # Load the ONNX model using Marabou
    network = Marabou.read_onnx(model_file_path)

    # Identify input and output variables from the network
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]

    # Read high fatigue data from CSV and extract mean values
    high_fatigue = pd.read_csv(csv_file_path)
    mean_values = high_fatigue['mean'][:-1].values
    min_values = high_fatigue['min'][:-1].values
    max_values = high_fatigue['max'][:-1].values

    # Initial range and step size setup
    initial_range = [0.01] * FEATURESNUM
    step_size = high_fatigue['std'][:-1].values * 0.01

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
            print("Iteration stopped at:", ITERATION)
            break

        # Define the output conditions for the model
        mf.define_output_conditions(network, outputVars, 2)

        # Solve the model with current configuration
        result = network.solve(verbose=True, options=options)
        status, values, stats = result

        # Check the results and update the range if necessary
        unsat, initial_range = mf.check_results(status, initial_range, step_size)

        # Handle SAT and UNSAT cases
        if status == "sat":
            print("Solution found!")
            print("Iteration stopped at:", ITERATION)

            min_unsat_range = initial_range.copy()
            range_bounds = [(mean_val - range_val, mean_val + range_val) for mean_val, range_val in
                            zip(mean_values, min_unsat_range)]
            print("最小 UNSAT 范围的界限:", range_bounds)
            # 可以选择将界限写入文件
            mf.write_values_to_csv(range_bounds, 'baseline_range_bounds.csv', __file__)


            output_values = [values[key] for key in range(10)]
            print(output_values)

            # Check if output values are within the high fatigue range
            if is_within_range(np.array(output_values), mean_values, high_fatigue['std'][:-1].values):
                plot_fatigue_distribution(high_fatigue, output_values)
            else:
                print("Output values are not within the high fatigue range.")
            mf.write_values_to_csv(values, 'top10_values.csv', __file__)
            unsat = False
        elif status == "unsat":
            print("No solution found in this iteration.")
            print("Time elapsed:", stats.getTotalTimeInMicro())

            # Update for the next iteration
            min_unsat_range = initial_range.copy()

    # Additional output handling as needed


if __name__ == "__main__":
    main()
