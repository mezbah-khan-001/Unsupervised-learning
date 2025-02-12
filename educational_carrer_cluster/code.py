# Hello world 
# LET'S code the program ---> 
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import time, os, sys, functools
from pathlib import Path 
from icecream import ic

# LET'S build the program --> 
data_path = Path('/content/education_career_success.csv')  
if data_path.exists():
    data = pd.read_csv(data_path)
    ic(f"Data loaded successfully from: {data_path}")
else:
    raise FileNotFoundError(f'This file path {data_path} does not exist. Please check.')

# LET'S check the information and structure --> 
ic(data.info())  # 5000 entries and float64(3), int64(12), object(5)
ic(data.isnull().sum())  # NO NaN values in data 
ic(data.describe())  # No outliers 

# LET'S build the model ---> 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Encode categorical features
encoder = LabelEncoder()
output_01 = pd.DataFrame(index=data.index)
data_columns_01 = data.select_dtypes(include=['object']).columns
for col in data_columns_01: 
    output_01[col] = encoder.fit_transform(data[col])
    ic(output_01.head(5))

# Scale numerical features
scaler = StandardScaler()
data_columns_02 = data.select_dtypes(include=['int64', 'float64']).columns
output_02 = pd.DataFrame(scaler.fit_transform(data[data_columns_02]), 
                         columns=data_columns_02, 
                         index=data.index)
ic(output_02.head(5))

# Combine encoded + scaled data
output_03 = pd.concat([output_01, output_02], axis=1)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust hyperparameters as needed
output_03['Cluster'] = dbscan.fit_predict(output_03)

# Visualize clusters using PCA (reduce dimensions)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(output_03.drop(columns=['Cluster']))

plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=output_03['Cluster'], palette='viridis')
plt.title("DBSCAN Clustering Visualization")
plt.show()

# Save processed dataset
output_03.to_csv('Educational_dataset_with_clusters.csv', index=False)

# Display clusters
ic(output_03['Cluster'].value_counts())
