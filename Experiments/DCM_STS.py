import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import configparser
from maraboupy import Marabou
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from Scripts.my_deco import running_time, debug, suppress_error
from Scripts import my_func as mf

# Constants
CONFIG_FILE_PATH = 'config.ini'

def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def load_and_preprocess_data(csv_path):
    feature_names = pd.read_csv('/home/adam/FurtherResearch/Dataset/STS/Data_Names.csv')
    header = feature_names['features'].tolist()
    dataset = pd.read_csv(csv_path)
    dataset.columns = header
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1] - 1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=dataset.columns[:-1])
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled.round(4), y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X_scaled

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)  # Convert predictions to scalar class labels
    y_test_argmax = np.argmax(y_test, axis=1)  # Convert true labels from one-hot to scalar class labels
    return confusion_matrix(y_test_argmax, y_pred), y_pred, y_test_argmax

def setup_and_solve_marabou(network_path, data_points):
    network = Marabou.read_onnx(network_path)
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]
    mf.define_output_conditions(network, outputVars, 2)

    solutions_indices = []
    counter = 0

    for index, row in data_points.iterrows():
        for i, value in enumerate(row):
            value = float(value)
            network.setUpperBound(inputVars[i], value)
            network.setLowerBound(inputVars[i], value)

        result = network.solve()
        status, values, stats = result
        if status == "sat":
            print(f"Solution found for data point {index}")
            solutions_indices.append(index)
            counter += 1
    print(f"Number of solutions found: {counter}")
    return data_points.loc[solutions_indices]

def detect_outliers_dynamic_threshold(large_dataset, small_dataset, lower_percentile=15, upper_percentile=85):
    """
    使用大数据集中马氏距离的动态百分位数作为阈值，检测小数据集中的异常点，并返回动态阈值。

    参数:
    - large_dataset: 大数据集的DataFrame。
    - small_dataset: 小数据集的DataFrame。
    - lower_percentile: 用于确定下界异常值阈值的百分位数，默认为15。
    - upper_percentile: 用于确定上界异常值阈值的百分位数，默认为85。

    返回:
    - outliers: 检测到的异常点的DataFrame。
    - lower_dynamic_threshold: 下界动态阈值。
    - upper_dynamic_threshold: 上界动态阈值。
    - small_distances: 小数据集中每个点的马氏距离数组。
    """
    # 计算大数据集的均值向量和协方差矩阵
    mean_vector = large_dataset.mean(axis=0)
    cov_matrix = large_dataset.cov()
    cov_matrix_inv = inv(cov_matrix)

    # 计算大数据集中每个点的马氏距离
    large_distances = np.array([mahalanobis(row, mean_vector, cov_matrix_inv) for index, row in large_dataset.iterrows()])

    # 计算动态阈值
    lower_dynamic_threshold = np.percentile(large_distances, lower_percentile)
    upper_dynamic_threshold = np.percentile(large_distances, upper_percentile)

    # 计算小数据集中每个点的马氏距离
    small_distances = np.array([mahalanobis(row, mean_vector, cov_matrix_inv) for index, row in small_dataset.iterrows()])

    # 标识超过动态阈值的异常点
    outliers_mask = np.logical_and(small_distances < upper_dynamic_threshold, small_distances > lower_dynamic_threshold)
    outliers = small_dataset[outliers_mask].copy()
    outliers['Mahalanobis_Distance'] = small_distances[outliers_mask]

    # 输出结果
    if outliers.empty:
        print(f"No outliers detected in the small dataset with the dynamic thresholds set at the {lower_percentile}th to {upper_percentile}th percentiles.")
    else:
        print(f"Hazard cases detected in the confusion matrix with the dynamic thresholds set at the {lower_percentile}th to {upper_percentile}th percentiles:")

    return outliers, lower_dynamic_threshold, upper_dynamic_threshold, small_distances


def main(config_file_path, model_file_path=None, model_with_softmax_file_path=None, dataset_file_path=None):
    config = read_config(config_file_path)

    model_for_CM = model_with_softmax_file_path or config['Model']['model_with_softmax_file_path']
    model_for_marabou = model_file_path or config['Model']['model_file_path']
    dataset_file_path = dataset_file_path or config['Dataset']['dataset_file_path']

    X_train, X_test, y_train, y_test, X_scaled = load_and_preprocess_data(dataset_file_path)
    model = tf.keras.models.load_model(model_for_CM)
    cm, y_pred, y_test_scalar = predict_and_evaluate(model, X_test, y_test)  # Adjusted function usage

    # Correct usage of y_test_scalar instead of y_test
    high_medium_indices = [i for i, (true, pred) in enumerate(zip(y_test_scalar, y_pred)) if true == 2 and pred == 1]
    high_medium_features = X_scaled.iloc[high_medium_indices, :].round(4)

    solutions_df = setup_and_solve_marabou(model_for_marabou, high_medium_features)

    print(f"Confusion matrix:\n{cm}")
    print('Hazards detected in the confusion matrix:')
    print(high_medium_features)

    print('Marabou detected hazards within CM:')
    print(f"Total solutions found: {len(solutions_df)}")
    print(solutions_df)

    # 假设 Baseline_dataset 和 Baseline_ce 已经是两个加载好的DataFrame
    outliers_result, lower_threshold, upper_threshold, distances =detect_outliers_dynamic_threshold(
        X_scaled.round(4), high_medium_features, lower_percentile=30, upper_percentile=70)

    print(f"Dynamic threshold：{lower_threshold} to {upper_threshold}")
    if not outliers_result.empty:
        print(outliers_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and evaluate exoskeleton data.')
    parser.add_argument('--config', type=str, default=CONFIG_FILE_PATH, help='Path to the configuration file.')
    parser.add_argument('--model_with_softmax_file_path', type=str, help='Path to the model file for CM with softmax layer.')
    parser.add_argument('--model_file_path', type=str, help='Path to the ONNX model file for Marabou.')
    parser.add_argument('--dataset_file_path', type=str, help='Path to the dataset file.')

    args = parser.parse_args()

    main(args.config, args.model_file_path, args.model_with_softmax_file_path, args.dataset_file_path)
