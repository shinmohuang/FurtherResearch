from maraboupy import Marabou
import pandas as pd

from Experiments import Scripts


# Main function
@Scripts.my_deco.running_time
def main():
    file_name = '/Baseline/Baseline/model_without_softmax.onnx'
    network = Marabou.read_onnx(file_name)

    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]

    high_fatigue = pd.read_csv('/home/adam/FurtherResearch/Experiments/Baseline/Statistic_analysis/dataset_statistics_fatigue_level_2.csv')
    mean_values = high_fatigue['mean'][:-1].values

    initial_range = [0.01] * 63
    step_size = high_fatigue['std'][:-1].values * 0.1

    options = Marabou.createOptions(numWorkers=20, initialTimeout=5, initialSplits=100, onlineSplits=100,
                                    timeoutInSeconds=1800, timeoutFactor=1.5,
                                    verbosity=2, snc=True, splittingStrategy='auto',
                                    sncSplittingStrategy='auto', restoreTreeStates=False,
                                    splitThreshold=20, solveWithMILP=True, dumpBounds=False)
    sat_counter = 0  # Initialize sat counter
    unsat = True
    while unsat:
        network = Marabou.read_onnx(file_name)
        Scripts.my_func.set_input_range(network, inputVars, mean_values, initial_range)
        Scripts.my_func.define_output_conditions(network, outputVars, 2)
        result = network.solve(verbose=True, options=options)
        status, values, stats = result
        unsat, initial_range = Scripts.my_func.check_results(status, initial_range, step_size)
        if status == "sat":
            print("Solution found!")
        elif status == "unsat":
            print("No solution found.")
            print("Time:", stats.getTotalTimeInMicro())


    range = [ir - ss for ir, ss in zip(initial_range, step_size)]
    print(f"最小 UNSAT 范围: {range}")
    Scripts.my_func.write_values_to_csv(range, 'baseline_range.csv')
    Scripts.my_func.write_values_to_csv(values, 'baseline_values.csv')

if __name__ == "__main__":
    main()