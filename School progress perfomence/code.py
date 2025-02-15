# 🚀 Import Libraries
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import os, sys, time, functools
from icecream import ic 
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN

# 📂 Load Dataset
data_path = Path('/content/school_data.xlsx')
if data_path.exists(): 
    data = pd.read_excel(data_path)
    ic('Data loaded successfully...')
else: 
    raise FileNotFoundError(f'This file path {data_path} does not exist...')

# 🛠️ Data Inspection
ic(data.info())  
ic(data.isnull().sum())  
ic(data.describe())  

# 🔍 Detect Outliers Function
def detect_outliers(data):
    outlier_columns = []
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        
        if not outliers.empty:  
            outlier_columns.append(col)
    
    return outlier_columns

outlier_cols = detect_outliers(data)
ic("Columns with outliers:", outlier_cols) 

# 📊 Boxplot for Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=data)  
plt.xticks(rotation=90) 
plt.title('Box Plot of All Columns (Outliers Detection)')
plt.show()

plt . figure(figsize=(5,10))
sns . boxplot(y='ZIP CODE'  , data = data)
plt . title(label='This is ZIP CODE outliers')
plt . grid(True)
plt . show()

plt . figure(figsize=(5,10))
sns . boxplot(y='LONGITUDE'  , data = data)
plt . title(label='This is LONGITUDE outliers')
plt . grid(True)
plt . show()


plt . figure(figsize=(5,10))
sns . boxplot(y='LATITUDE'  , data = data)
plt . title(label='This is LATITUDE outliers')
plt . grid(True)
plt . show()


# 🔄 Feature Scaling
scaler_float, scaler_int = MinMaxScaler(), StandardScaler()
float_cols, int_cols = data.select_dtypes(include=['float64']).columns, data.select_dtypes(include=['int64']).columns

data[float_cols] = scaler_float.fit_transform(data[float_cols])
data[int_cols] = scaler_int.fit_transform(data[int_cols])

# 💾 Save Cleaned Dataset
data.to_csv("school_progress_dataset.csv", index=False)
df = pd.read_csv("school_progress_dataset.csv")
ic(df.describe())
ic(df.info())

# 🤖 Apply DBSCAN Clustering
features = ['LATITUDE', 'LONGITUDE']  
X = data[features].copy()

# 🏗️ Standardizing Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🚀 Train DBSCAN Model
dbscan = DBSCAN(eps=0.3, min_samples=6)
data['Cluster'] = dbscan.fit_predict(X_scaled)
ic(set(data['Cluster']))  

# 🎨 Visualizing Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['LONGITUDE'], y=data['LATITUDE'], hue=data['Cluster'], palette='viridis', s=100)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clustering of Schools')
plt.legend(title='Cluster')
plt.show()
    
    # the Cluster 5 School Progress Performance Model is not only 
    # a reflection of a school’s current state but also a blueprint for continuous growth,
    # aiming to provide every student with a high-quality, inclusive, and forward-thinking educational experience.
    #THE Mdoel have total 5 cluster ... 
    
