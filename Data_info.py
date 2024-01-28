from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import numpy as np
import pandas as pd
dataset=pd.read_csv("industrial_dataset.csv")
# Checking for missing values
missing_values = dataset.isnull().sum()

# Checking for duplicates
duplicates = dataset.duplicated().sum()

# Basic statistics for detecting potential outliers
basic_stats = dataset.describe()

print(missing_values, duplicates, basic_stats)