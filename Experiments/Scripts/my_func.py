from maraboupy import Marabou, MarabouCore, MarabouUtils
import os
import csv
import numpy as np


def write_values_to_csv(values, file_path):
    """
    Write values to a csv file.

    Args:
        values: The values to be written.
        file_path: The complete file path of the CSV file.

    Returns:
        None
    """
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Check if values is a dictionary or numpy array and convert to list
        if isinstance(values, dict):
            values = list(values.values())
        elif isinstance(values, np.ndarray):
            values = values.tolist()

        writer.writerow(values)

    print(f"Values written to {file_path}")




def set_input_range(network, inputVars, mean_values, initial_range):
    for i, mean_val in enumerate(mean_values):
        network.setLowerBound(inputVars[i], (mean_val - initial_range[i]).round(4))
        network.setUpperBound(inputVars[i], (mean_val + initial_range[i]).round(4))


# def define_output_conditions(network, outputVars, desired_output_class):
#     for i in range(len(outputVars)):
#         if i != desired_output_class:
#             network.addInequality([outputVars[0][i], outputVars[0][desired_output_class]], [1, -1], 0)

# def define_output_conditions(network, outputVars, desired_output_class):
#     for i in range(len(outputVars)):
#         if i != desired_output_class:
#             # Ensure that each class other than the desired_output_class
#             # has a greater output value than the desired_output_class.
#             # This is achieved by setting outputVars[0][i] - outputVars[0][desired_output_class] > 0
#             network.addInequality([outputVars[0][i], outputVars[0][desired_output_class]], [1, -1], 1)
#
# from maraboupy import MarabouUtils, MarabouCore

def define_output_conditions(network, outputVars, non_dominant_class):
    """
    Ensure that the specified class is not the dominant (largest output) class.

    Args:
        network (MarabouNetwork): The Marabou network object.
        outputVars (list of int): List of output variable indices.
        non_dominant_class (int): Index of the class to ensure is not dominant.
    """
    num_classes = len(outputVars)
    for i in range(num_classes):
        if i != non_dominant_class:
            # Create inequality: outputVars[non_dominant_class] - outputVars[i] < 0
            vars = [outputVars[0][non_dominant_class], outputVars[0][i]]
            coeffs = [1, -1]
            network.addInequality(vars, coeffs, -1)


# def add_non_zero_input_constraint(network, inputVars):
#     """
#     Adds a constraint to the network to ensure that not all inputs can be zero.
#
#     Args:
#         network (MarabouNetwork): The Marabou network object.
#         inputVars (list): List of input variables.
#
#     Returns:
#         None
#     """
#     disjunction = []  # 用于存储所有的不等式
#
#     for var in inputVars:
#         # 变量大于零的不等式
#         greater_than_zero = MarabouCore.Equation(MarabouCore.Equation.GE)
#         greater_than_zero.addAddend(1, var)
#         greater_than_zero.setScalar(0.1)  # 可以调整为适合您的应用的小正数
#
#         # 变量小于零的不等式
#         less_than_zero = MarabouCore.Equation(MarabouCore.Equation.LE)
#         less_than_zero.addAddend(1, var)
#         less_than_zero.setScalar(-0.1)  # 可以调整为适合您的应用的小负数
#
#         # 添加到“或”约束中
#         disjunction.append([greater_than_zero, less_than_zero])
#
#     # 将“或”约束添加到网络
#     network.addDisjunctionConstraint(disjunction)

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
    for var in inputVars:
        value = values[var.item()]

        # Less than constraint
        eq1 = MarabouCore.Equation(MarabouCore.Equation.LE)
        eq1.addAddend(1, var)
        eq1.setScalar(value - 0.05)

        # Greater than constraint
        eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq2.addAddend(1, var)
        eq2.setScalar(value + 0.05)

        # Add disjunction
        network.addDisjunctionConstraint([[eq1], [eq2]])

        # print(f"Blocking var {var.item()}: not in [{value - 0.05}, {value + 0.05}]")

