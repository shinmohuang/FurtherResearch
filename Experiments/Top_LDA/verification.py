from maraboupy import Marabou
from Scripts.my_deco import running_time, debug, suppress_error
from Scripts import my_func as mf
import pandas as pd


# Main function
@running_time
@suppress_error
def main():
    file_name = '/home/adam/FurtherResearch/Model/Top_LDA/exo_model_top10_without_softmax.onnx'
    network = Marabou.read_onnx(file_name)

    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]

    high_fatigue = pd.read_csv(
        '/home/adam/FurtherResearch/Experiments/Top_LDA/dataset_statistics_mapped_fatigue_level_high.csv')
    mean_values = high_fatigue['mean'][:-1].values

    initial_range = [0.01] * 10
    step_size = high_fatigue['std'][:-1].values * 0.01

    options = Marabou.createOptions(numWorkers=20, initialTimeout=5, initialSplits=100, onlineSplits=100,
                                    timeoutInSeconds=1800, timeoutFactor=1.5,
                                    verbosity=2, snc=True, splittingStrategy='auto',
                                    sncSplittingStrategy='auto', restoreTreeStates=False,
                                    splitThreshold=20, solveWithMILP=True, dumpBounds=False)
    sat_counter = 0  # Initialize sat counter
    unsat = True
    while unsat:
        network = Marabou.read_onnx(file_name)
        mf.set_input_range(network, inputVars, mean_values, initial_range)
        mf.define_output_conditions(network, outputVars, 2)
        result = network.solve(verbose=True, options=options)
        status, values, stats = result
        unsat, initial_range = mf.check_results(status, initial_range, step_size)
        if status == "sat":
            print("Solution found!")
        elif status == "unsat":
            print("No solution found.")
            print("Time:", stats.getTotalTimeInMicro())

    range = [ir - ss for ir, ss in zip(initial_range, step_size)]
    print(f"最小 UNSAT 范围: {range}")
    mf.write_values_to_csv(range, 'top10_range.csv', __file__)
    mf.write_values_to_csv(values, 'top10_values.csv', __file__)


if __name__ == "__main__":
    main()
