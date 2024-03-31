import argparse
import configparser
import pandas as pd
from maraboupy import Marabou
from maraboupy.MarabouNetworkONNX import MarabouNetworkONNX
from Scripts.my_deco import running_time, suppress_error
from Scripts import my_func as mf

# Constants
DEFAULT_CONFIG_FILE = 'config.ini'

def get_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def load_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    return data['mean'][:-1].values.round(4), data['min'][:-1].values.round(4), data['max'][:-1].values.round(4)

def update_progress_bar(current_value, min_value, max_value):
    # 计算当前值在最大最小值范围内的相对位置
    progress = ((current_value - min_value) / (max_value - min_value)) * 100
    # 限制进度在0%到100%之间
    progress = max(0, min(100, progress))
    progress_bar = f"[{'#' * int(progress // 10)}{'-' * (10 - int(progress // 10))}] {progress:.2f}%"
    print(f"\r{progress_bar}", end='')

def solve_model(model_file_path, inputVars, outputVars, min_values, max_values, log_file_path, counterExample_to_file):
    options = Marabou.createOptions(snc=True, numWorkers=8)
    solutions, unsat, solution_count = [], False, 0
    min_value, max_value = min_values[0], max_values[0]  # 取第一个特征的最小最大值

    while not unsat:
        network = Marabou.read_onnx(model_file_path)
        for solution in solutions:
            mf.block_solution(network, solution, inputVars)
        for i, var in enumerate(inputVars):
            network.setLowerBound(var, min_values[i])
            network.setUpperBound(var, max_values[i])
        mf.define_output_conditions(network, outputVars, 2)
        result = network.solve(filename=log_file_path, options=options)
        status, values, _ = result
        if status == "sat":
            solutions.append(values)
            solution_count += 1  # Increment the solution count
            # print("inputVars structure:", inputVars)
            # print("Example value from 'values' dict:", list(values.items())[0])

            current_value = values[inputVars[0]]  # 假设values字典中键为输入变量ID
            update_progress_bar(current_value, min_value, max_value)
            print(f"Solution {solution_count} found.")# Print the current solution count
            mf.write_values_to_csv(values, counterExample_to_file)
        else:
            unsat = True
            print("\nNo more solutions. Progress: 100%")
    return solutions

@running_time
@suppress_error
def main(config_file, model_file_path=None, csv_file_path=None, log_file_path=None, counterExample_to_file=None):
    config = get_config(config_file)
    model_file_path = model_file_path or config['Model']['model_file_path']
    csv_file_path = csv_file_path or config['Dataset']['csv_file_path']
    log_file_path = log_file_path or config['Logging']['log_file_path']
    counterExample_to_file = counterExample_to_file or config['Logging']['counterExample_to_file']
    # min_unsat_range_to_file = min_unsat_range_to_file or config['Logging']['min_unsat_range_to_file']

    network = Marabou.read_onnx(model_file_path)
    inputVars, outputVars = network.inputVars[0][0], network.outputVars[0]
    mean_values, min_values, max_values = load_data(csv_file_path)
    solutions = solve_model(model_file_path, inputVars, outputVars, min_values, max_values, log_file_path, counterExample_to_file)
    if solutions:
        print(f"Total solutions found: {len(solutions)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Marabou model verification.')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_FILE, help='Path to the configuration file.')
    parser.add_argument('--model_file_path', type=str, help='Path to the ONNX model file.')
    parser.add_argument('--csv_file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('--log_file_path', type=str, help='Path to the log file.')
    parser.add_argument('--counterExample_to_file', type=str, help='Path to the counterExample file.')
    # parser.add_argument('--min_unsat_range_to_file', type=str, help='Path to the min_unsat_range file.')

    args = parser.parse_args()
    main(args.config, args.model_file_path, args.csv_file_path, args.log_file_path, args.counterExample_to_file)
