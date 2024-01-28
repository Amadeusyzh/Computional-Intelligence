import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from pyswarm import pso

# Step 1: Load the dataset
dataset = pd.read_csv('industrial_dataset.csv')
X = dataset.drop('Target', axis=1)  # Assuming 'Target' is the name of the target column
y = dataset['Target']

# Normalize the feature data
X_normalized = (X - X.mean()) / X.std()

# Step 2: Define the PSO algorithm for feature selection
def feature_selection_pso(X, y):
    # Number of features
    n_features = X.shape[1]

    # Define the objective function for PSO to minimize
    def objective_function(selected_features_mask):
        # Convert to integer boolean mask
        selected_features_mask = selected_features_mask.astype(bool)
        # Select the features
        X_selected = X_normalized.iloc[:, selected_features_mask]
        # Evaluate model with cross-validation
        scores = cross_val_score(LinearRegression(), X_selected, y, cv=5)
        # Objective is to maximize the score, so minimize negative score
        return -scores.mean()

    # Define the PSO arguments
    lb = np.zeros(n_features)  # Lower bounds
    ub = np.ones(n_features)   # Upper bounds

    # Run PSO
    xopt, fopt = pso(objective_function, lb, ub, swarmsize=50, maxiter=1000)

    # Return the selected features and the objective function value
    selected_features_mask = xopt > 0.5  # Assuming threshold of 0.5
    return selected_features_mask, -fopt

# Step 3: Run PSO and get the best feature subset
best_features_mask, best_score = feature_selection_pso(X_normalized, y)
selected_features_indices = np.where(best_features_mask > 0.5)[0]  # Get indices of selected features
print("Selected Features Indices:", selected_features_indices)
print("Best Score:", best_score)

# Step 4: Train and evaluate the final model
X_selected_final = X_normalized.iloc[:, best_features_mask.astype(bool)]
X_train, X_test, y_train, y_test = train_test_split(X_selected_final, y, test_size=0.2, random_state=42)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Final MSE:", mse)