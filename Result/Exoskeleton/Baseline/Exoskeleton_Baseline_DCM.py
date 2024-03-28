import tensorflow as tf
import pandas as pd
import numpy as np
from maraboupy import Marabou
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

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



# Read the dataset CSV file
dataset = pd.read_csv('../../../Dataset/Exoskeleton/Original.csv')

X = dataset.iloc[:, :-2]
y = dataset.iloc[:, -2] - 1

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=dataset.columns[:-2])
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled.round(4), y, test_size=0.2, random_state=42)

model = tf.keras.models.load_model('../../../Model/Exoskeleton/Baseline/exo_model')

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_test, y_pred)

high_medium_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true == 2 and pred == 1]
high_medium_features = X_scaled.iloc[high_medium_indices, :].round(4)

# Load the network
network = Marabou.read_onnx('/home/adam/FurtherResearch/Model/Exoskeleton/Baseline/exo_model_without_softmax.onnx')

# Get the input and output variable numbers, assuming a single input and output
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0]

# Set the input constraints for the particular data point you want to test
data_point = high_medium_features  # Replace with the values of your data point

define_output_conditions(network, outputVars, 2)

counter = 0

# 对 data_point 的每一行进行迭代
for index, row in data_point.iterrows():
    # 对每个输入变量设置上下边界
    for i, value in enumerate(row):
        # 确保 value 是 float 类型
        value = float(value)
        # 设置第 i 个输入变量的上下边界
        network.setUpperBound(inputVars[i], value)
        network.setLowerBound(inputVars[i], value)

    # 求解网络
    result = network.solve()

    status, values, stats = result
    if status == "sat":
        print(f"Solution found for data point {index}")
        counter += 1

print(f"Confusion matrix:\n{cm}")
print(high_medium_features)
print(f"Number of solutions found: {counter}")


def detect_outliers_dynamic_threshold(large_dataset, small_dataset, percentile=95):
    """
    使用大数据集中马氏距离的动态百分位数作为阈值，检测小数据集中的异常点，并返回动态阈值。

    参数:
    - large_dataset: 大数据集的DataFrame。
    - small_dataset: 小数据集的DataFrame。
    - percentile: 用于确定异常值阈值的百分位数，例如95或99，默认为95。

    返回:
    - outliers: 检测到的异常点的DataFrame。
    - dynamic_threshold: 动态阈值。
    - small_distances: 小数据集中每个点的马氏距离数组。
    """
    # 计算大数据集的均值向量和协方差矩阵
    mean_vector = large_dataset.mean(axis=0)
    cov_matrix = large_dataset.cov()
    cov_matrix_inv = inv(cov_matrix)

    # 计算大数据集中每个点的马氏距离
    large_distances = np.array(
        [mahalanobis(row, mean_vector, cov_matrix_inv) for index, row in large_dataset.iterrows()])

    # 计算动态阈值
    dynamic_threshold = np.percentile(large_distances, percentile)

    # 计算小数据集中每个点的马氏距离
    small_distances = np.array(
        [mahalanobis(row, mean_vector, cov_matrix_inv) for index, row in small_dataset.iterrows()])

    # 标识超过动态阈值的异常点
    outliers_mask = small_distances > dynamic_threshold
    outliers = small_dataset[outliers_mask].copy()
    outliers['Mahalanobis_Distance'] = small_distances[outliers_mask]

    # 输出结果
    if outliers.empty:
        print(
            f"No outliers detected in the small dataset with the dynamic threshold set at the {percentile}th percentile.")
    else:
        print(
            f"Outliers detected in the small dataset with the dynamic threshold set at the {percentile}th percentile:")

    return outliers, dynamic_threshold, small_distances


# 使用示例
# 假设 Baseline_dataset 和 Baseline_ce 已经是两个加载好的DataFrame
outliers_result, threshold, distances = detect_outliers_dynamic_threshold(X_scaled.round(4), data_point, percentile=99)
print("Dynamic threshold：", threshold)
if not outliers_result.empty:
    print(outliers_result)