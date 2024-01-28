from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import numpy as np
import pandas as pd
dataset=pd.read_csv("industrial_dataset.csv")
# Prepare the data
X = dataset.iloc[:, 1:-1]  # all features except the target
y = dataset.iloc[:, -1]   # the target variable

# Initialize models
linear_model = LinearRegression()
svr_model = SVR()

# Perform cross-validation and compute the mean score
cv_scores_linear = cross_val_score(linear_model, X, y, cv=5)
cv_scores_svr = cross_val_score(svr_model, X, y, cv=5)

mean_score_linear = np.mean(cv_scores_linear)
mean_score_svr = np.mean(cv_scores_svr)

print("linear_Regression_model:",mean_score_linear, "SVR_model:", mean_score_svr)

# Create MSE scorer
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform cross-validation and compute the mean score
cv_scores_linear = cross_val_score(linear_model, X, y, cv=5, scoring=mse_scorer)
cv_scores_svr = cross_val_score(svr_model, X, y, cv=5, scoring=mse_scorer)

# Calculate mean MSE scores
mean_mse_linear = -np.mean(cv_scores_linear)  # Negating the value because the scorer returns negative MSE
mean_mse_svr = -np.mean(cv_scores_svr)       # Negating the value because the scorer returns negative MSE

print("Mean MSE for Linear Regression model:", mean_mse_linear)
print("Mean MSE for SVR model:", mean_mse_svr)