import pandas as pd

# Read the feature names CSV file
feature_names = pd.read_csv(r'/Dataset/Feature Name.csv')

# Create a list of feature names
header_list = feature_names['Feature'].tolist()

# Read the dataset CSV file
dataset = pd.read_csv(r'/Dataset/Dataset.csv', names=header_list)

# Print the first few rows of the dataset and feature names
print(dataset.head())

# Print the shape of the dataset
print(dataset.shape)

# Print the column names of the dataset
print(dataset.columns)

# Visualize the dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Replace 'Class' with the actual target variable name in your dataset
target_variable = 'Fatigue_level'

# Create a countplot
sns.countplot(x=target_variable, data=dataset, palette='hls')
plt.show()

# Create a histogram
dataset.hist(bins=20, figsize=(20, 20))
plt.show()

# Calculate the number of rows and columns needed for the layout
n_cols = int(dataset.shape[1] ** 0.5)
n_rows = n_cols if n_cols * n_cols == dataset.shape[1] else n_cols + 1

# Create a boxplot
dataset.plot(kind='box', subplots=True, layout=(n_rows, n_cols), sharex=False, sharey=False, figsize=(20, 20))
plt.show()

# Create a correlation matrix
corrmat = dataset.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# Create a scatterplot
sns.set()
cols = header_list
sns.pairplot(dataset[cols], size=2.5)
plt.show()
