import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('../../../Dataset/STS/Reduced_STS.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1] - 1

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = tf.keras.models.load_model('../Model/STS/FeatureReduced/sts_model_top10')

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_test, y_pred)

high_medium_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true == 2 and pred == 1]
high_medium_features = X.iloc[high_medium_indices, :]

from maraboupy import Marabou
# Load the network
network = Marabou.read_onnx('../../../Model/STS/FeatureReduced/sts_model_top10_without_softmax.onnx')

# Get the input and output variable numbers, assuming a single input and output
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0]

# Set the input constraints for the particular data point you want to test
data_point = high_medium_features  # Replace with the values of your data point

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

print(f"Number of solutions found: {counter}")