from maraboupy import Marabou, MarabouCore
import pandas as pd
import os
import numpy as np
import csv

def write_values_to_csv(values, filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_directory, filename)

    with open(full_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Check if values is a dictionary
        if isinstance(values, dict):
            values = list(values.values())
        # If it's a numpy array, convert it to a list
        elif isinstance(values, np.ndarray):
            values = values.tolist()

        writer.writerow(values)

    print(f"Values written to {full_path}")

def set_input_range(network, inputVars, mean_values, initial_range):
    for i, mean_val in enumerate(mean_values):
        network.setLowerBound(inputVars[i], mean_val - initial_range[i])
        network.setUpperBound(inputVars[i], mean_val + initial_range[i])

def define_output_conditions(network, outputVars, desired_output_class):
    for i in range(len(outputVars)):
        if i != desired_output_class:
            network.addInequality([outputVars[0][i], outputVars[0][desired_output_class]], [1, -1], 0)



def check_results(status, initial_range, step_size):
    if status == "unsat":
        for i in range(len(initial_range)):
            initial_range[i] += step_size[i]
        return True, initial_range
    else:
        return False, initial_range

def block_solution(network, values, inputVars):
    """
    Adds constraints to the network to block the current solution.

    Args:
    network (MarabouNetwork): The Marabou network object.
    values (dict): The current solution values.
    inputVars (list): List of input variables.
    """
    for var in inputVars:
        # Create a constraint that the variable is less than the current solution
        eq1 = MarabouCore.Equation(MarabouCore.Equation.LE)
        eq1.addAddend(1, var)
        eq1.setScalar(values[var.item()] - 0.05)

        # Create a constraint that the variable is greater than the current solution
        eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq2.addAddend(1, var)
        eq2.setScalar(values[var.item()] + 0.05)

        # Add the disjunction of the two constraints to the network
        network.addDisjunctionConstraint([[eq1], [eq2]])


# Main function
def main():
    file_name = 'model_without_softmax.onnx'
    network = Marabou.read_onnx(file_name)

    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]

    high_fatigue = pd.read_csv('statistic_analysis/dataset_statistics_fatigue_level_2.csv')
    mean_values = high_fatigue['mean'][:-1].values

    initial_range = [0.01] * 63
    step_size = high_fatigue['std'][:-1].values * 0.1

    options = Marabou.createOptions(numWorkers=20, initialTimeout=5, initialSplits=100, onlineSplits=100,
                                    timeoutInSeconds=1800, timeoutFactor=1.5,
                                    verbosity=2, snc=True, splittingStrategy='auto',
                                    sncSplittingStrategy='auto', restoreTreeStates=False,
                                    splitThreshold=20, solveWithMILP=True, dumpBounds=True)
    sat_counter = 0  # Initialize sat counter
    unsat = True
    while unsat:
        network = Marabou.read_onnx(file_name)
        set_input_range(network, inputVars, mean_values, initial_range)
        define_output_conditions(network, outputVars, 2)
        result = network.solve(verbose=True, options=options)
        status, values, stats = result
        unsat, initial_range = check_results(status, initial_range, step_size)
        if status == "sat":
            print("Solution found!")
            sat_counter += 1

            # 使用解的阻塞函数
            block_solution(network, values, inputVars)

            # 检查是否已达到所需的解数量
            if sat_counter >= 10:
                break


    range = [ir - ss for ir, ss in zip(initial_range, step_size)]
    print(f"最小 UNSAT 范围: {range}")
    write_values_to_csv(range, 'baseline_range.csv')
    write_values_to_csv(values, 'baseline_values.csv')

if __name__ == "__main__":
    main()