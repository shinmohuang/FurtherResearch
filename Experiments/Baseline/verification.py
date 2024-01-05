from maraboupy import Marabou
from Scripts.my_deco import running_time, debug, suppress_error
from Scripts import my_func as mf
import pandas as pd



# Main function
@running_time
def main():
    file_name = '/home/adam/FurtherResearch/Model/Baseline/model_without_softmax.onnx'
    network = Marabou.read_onnx(file_name)

    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]

    high_fatigue = pd.read_csv('/home/adam/FurtherResearch/Experiments/Baseline/Statistic_analysis/dataset_statistics_fatigue_level_2.csv')
    mean_values = high_fatigue['mean'][:-1].values

    initial_range = [0.01] * 63
    step_size = high_fatigue['std'][:-1].values * 0.1

    options = Marabou.createOptions(snc=True, splittingStrategy='auto',
                                    sncSplittingStrategy='auto', restoreTreeStates=True,
                                    splitThreshold=20, solveWithMILP=True, dumpBounds=True)
    # sat_counter = 0  # Initialize sat counter
    unsat = True
    min_unsat_range = 0
    ITERATION = 0
    while unsat:
        print(f"第 {ITERATION} 次迭代")
        ITERATION += 1
        network = Marabou.read_onnx(file_name)
        mf.set_input_range(network, inputVars, mean_values, initial_range)
        mf.add_non_zero_input_constraint(network, inputVars)
        mf.define_output_conditions(network, outputVars, 2)
        # network.saveQuery("baseline_query.txt")
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
        elif status == "unsat":
            print("No solution found.")
            print("Time:", stats.getTotalTimeInMicro())



    # range = [ir - ss for ir, ss in zip(initial_range, step_size)]
    # print(f"最小 UNSAT 范围: {range}")
    # mf.write_values_to_csv(range, 'baseline_range.csv', __file__)
    mf.write_values_to_csv(values, 'baseline_values.csv', __file__)

if __name__ == "__main__":
    main()