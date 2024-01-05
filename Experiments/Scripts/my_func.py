from maraboupy import Marabou, MarabouCore
import os
import csv
import numpy as np


def write_values_to_csv(values, filename, caller_file_path):
    """
    Write values to a csv file in the directory of the caller file.

    Args:
        values: The values to be written.
        filename: The name of the file to be written.
        caller_file_path: The file path of the caller.

    Returns:
        None
    """
    # Get the directory of the caller file
    caller_directory = os.path.dirname(os.path.abspath(caller_file_path))
    full_path = os.path.join(caller_directory, filename)

    with open(full_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Check if values is a dictionary or numpy array and convert to list
        if isinstance(values, dict):
            values = list(values.values())
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


def add_non_zero_input_constraint(network, inputVars):
    """
    Adds a constraint to the network to ensure that not all inputs can be zero.

    Args:
        network (MarabouNetwork): The Marabou network object.
        inputVars (list): List of input variables.

    Returns:
        None
    """
    disjunction = []  # 用于存储所有的不等式

    for var in inputVars:
        # 变量大于零的不等式
        greater_than_zero = MarabouCore.Equation(MarabouCore.Equation.GE)
        greater_than_zero.addAddend(1, var)
        greater_than_zero.setScalar(0.1)  # 可以调整为适合您的应用的小正数

        # 变量小于零的不等式
        less_than_zero = MarabouCore.Equation(MarabouCore.Equation.LE)
        less_than_zero.addAddend(1, var)
        less_than_zero.setScalar(-0.1)  # 可以调整为适合您的应用的小负数

        # 添加到“或”约束中
        disjunction.append([greater_than_zero, less_than_zero])

    # 将“或”约束添加到网络
    network.addDisjunctionConstraint(disjunction)

# @debug
def check_results(status, initial_range, step_size):
    """
    check the results of the verification

    Args:
        status: the status of the verification
        initial_range: the initial range of the input
        step_size: the step size of the input

    Returns:
        unsat: whether the verification is unsat
        initial_range: the updated initial range
    """
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