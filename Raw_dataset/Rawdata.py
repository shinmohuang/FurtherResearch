from scipy.io import loadmat
import pandas as pd

# Use raw string for the path
path = r'D:\OneDrive - University of Bristol\FurtherResearch\Raw_dataset\OneDrive_1_2023-11-5\S1\S1_Session1_EMG.mat'

# Load the .mat file
features_struct = loadmat(path)

# Convert the relevant data to a DataFrame
# Assuming 'data' is the key holding the main data
if 'data' in features_struct:
    df = pd.DataFrame(features_struct['data'])
    print(df)
else:
    print("No 'data' key found in the .mat file.")
