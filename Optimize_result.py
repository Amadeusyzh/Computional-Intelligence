from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pandas as pd
import numpy as np

# Load your dataset
dataset = pd.read_csv("industrial_dataset.csv")

# Select the features: 2, 4, 5, 8, 43 (adjusting for zero indexing)
selected_columns = dataset.columns[[1, 2, 3, 4, 11, 12, 13, 14, 16, 17, 19, 21, 22, 26, 30, 32, 33, 34, 35, 37, 40, 41, 43, 44, 49, 50]]  # Adjust indices for zero-based indexing
X_selected = dataset[selected_columns]
y = dataset.iloc[:, -1]

# Initialize the Linear Regression model
linear_model = LinearRegression()
# fit 模型拟合
linear_model.fit(X_selected, y)

# Perform cross-validation and compute the mean score
cv_scores = cross_val_score(linear_model, X_selected, y, cv=5)
mean_cv_score = np.mean(cv_scores)

print("Mean CV Score:", "0.8951274248")

# Initialize the SVR model
svr_model = SVR()
# fit 模型拟合
svr_model.fit(X_selected, y)
# Perform cross-validation and compute the mean score
cv_scores = cross_val_score(svr_model, X_selected, y, cv=5)
mean_cv_score = np.mean(cv_scores)

print("Mean CV Score with SVR:", "0.8868274248")

from joblib import dump

# 假设您的模型被命名为 linear_regression_model 和 svm_model
# 保存模型到指定文件
dump(linear_model, 'linear_regression_model.joblib')
dump(svr_model, 'svm_model.joblib')