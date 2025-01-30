# Hello wrold
   # Lets code the program with AI .........
      # Lets load the modules .......
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from icecream import ic

   # lets load the dataset --->
data_path = Path('/content/Europe_GDP.csv')
if data_path.exists() :
  df = pd.read_csv(data_path)
  ic('data load successfully.......')
else:
  raise FileNotFoundError(f'The file path {data_path}doest founded .....')
     # lets check the data -->
ic(df.head(5))
ic(df.info())   # There are 19 columns and 64 rows (float64(18), int64(1))
ic(df.isnull().sum())  #There are no NaN values in dataset ...
  #lets check the datas outliers ...
ic(df.describe())  #Every columns have an Outliers
 # lets create a functions that autometially remove outliers .......

def remove_outliers(df):
    for col in df.select_dtypes(include=['number']).columns:  # Process only numeric columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.drop(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index, inplace=True)
    ic("Outliers removed successfully!")

remove_outliers(df)  # This modifies 'df' directly
ic(df.describe())  # Check the modified dataset
   #The outliers is removed
   #Lets encode the data into bainary formate .....

   # Initialize MinMaxScaler
scaler = MinMaxScaler()

   # Select float64 columns
data_columns_01 = df.select_dtypes(include=['float64']).columns

   # Apply MinMax Scaling and replace original values
df[data_columns_01] = scaler.fit_transform(df[data_columns_01])

ic(df.head())  # Check the modified dataset

   #the dataset is encoded ......

from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Initialize DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=3)

# Fit DBSCAN to your dataset (excluding the integer column if needed)
cluster_labels = dbscan.fit_predict(df)

# Add cluster labels to the dataset
df['Cluster'] = cluster_labels

# Check the cluster distribution
print(df['Cluster'].value_counts())

# If your data is 2D or can be visualized:
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=cluster_labels, cmap='viridis')
plt.title("DBSCAN Clustering")
plt.show()

print(df['Cluster'].value_counts())
print(df['Cluster'].value_counts())

