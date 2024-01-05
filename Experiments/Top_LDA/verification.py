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
    # sat_counter = 0  # Initialize sat counter
    unsat = True
    min_unsat_range = None
    ITERATION = 0

    while unsat:
        print(f"第 {ITERATION} 次迭代")
        ITERATION += 1
        network = Marabou.read_onnx(file_name)
        mf.set_input_range(network, inputVars, mean_values, initial_range)
        mf.define_output_conditions(network, outputVars, 2)
        result = network.solve(verbose=True, options=options)
        status, values, stats = result
        unsat, initial_range = mf.check_results(status, initial_range, step_size)
        if status == "sat":
            min_unsat_range = initial_range.copy()
            range_bounds = [(mean_val - range_val, mean_val + range_val) for mean_val, range_val in
                            zip(mean_values, min_unsat_range)]
            print("最小 UNSAT 范围的界限:", range_bounds)
            # 可以选择将界限写入文件
            mf.write_values_to_csv(range_bounds, 'baseline_range_bounds.csv', __file__)
            print("Solution found!")
            unsat = False
        elif status == "unsat":
            min_unsat_range = initial_range.copy()
            print("No solution found.")
            print("Time:", stats.getTotalTimeInMicro())



    # print(f"最小 UNSAT 范围: {range}")
    # mf.write_values_to_csv(range, 'top10_range.csv', __file__)
    mf.write_values_to_csv(values, 'top10_values.csv', __file__)
    # 可以选择将界限写入文件
    # mf.write_values_to_csv(range_bounds, 'baseline_range_bounds.csv', __file__)

if __name__ == "__main__":
    main()
