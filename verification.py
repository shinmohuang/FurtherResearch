from maraboupy import Marabou, MarabouCore
import pandas as pd


# 加载模型和数据
file_name = 'model_without_softmax.onnx'
network = Marabou.read_onnx(file_name)

# 获取输入和输出变量
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0]

# 加载高疲劳统计数据
high_fatigue = pd.read_csv('statistic_analysis/dataset_statistics_fatigue_level_2.csv')
mean_values = high_fatigue['mean'][:-1].values

# 定义初始输入范围（以均值为中心的小范围）
initial_range= [0.01] * 63  # 初始范围
step_size = high_fatigue['std'][:-1].values * 0.1      # 每次迭代增加的范围

# options = Marabou.createOptions(numWorkers=20, initialTimeout=5, initialSplits=100, onlineSplits=100,
#                                     timeoutInSeconds=1800, timeoutFactor=1.5,
#                                     verbosity=2, snc=True, splittingStrategy='auto',
#                                     sncSplittingStrategy='auto', restoreTreeStates=False,
#                                     splitThreshold=20, solveWithMILP=True, dumpBounds=True)

# 迭代过程
unsat = True
while unsat:
    # 重置网络
    network = Marabou.read_onnx(file_name)

    # 设置输入范围
    for i, mean_val in enumerate(mean_values):
        network.setLowerBound(inputVars[i], mean_val - initial_range[i])
        network.setUpperBound(inputVars[i], mean_val + initial_range[i])

    # 定义输出条件
    desired_output_class = 2  # 高疲劳类别索引
    for i in range(len(outputVars)):
        if i != desired_output_class:
            network.addInequality([outputVars[0][i], outputVars[0][desired_output_class]], [1, -1], 0)

    # 运行验证
    # vals = network.solve(verbose=True,options=options)[0]
    vals = network.solve(verbose=True)[0]

    # 检查结果
    if vals == "unsat":
        # 如果是 UNSAT，增加输入范围
        for i in range(len(initial_range)):
            initial_range[i] += step_size[i]
    else:
        # 如果不是 UNSAT，停止迭代
        unsat = False

# 输出最小 UNSAT 范围
print(f"最小 UNSAT 范围: {initial_range - step_size}")