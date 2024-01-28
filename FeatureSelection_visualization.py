import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Particle:
    # ... (Particle class remains the same)
    def __init__(self, num_features):
        self.position = np.zeros(num_features, dtype=bool)
        self.position[:5] = True  # Initialize with 5 features selected
        np.random.shuffle(self.position)  # Shuffle to randomize selected features
        self.velocity = np.random.rand(num_features) - 0.5
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def update_velocity(self, global_best_position, w=0.5, c1=1, c2=2):
        r1, r2 = np.random.rand(2)
        v1 = (self.best_position != self.position).astype(int)
        v2 = (global_best_position != self.position).astype(int)
        self.velocity = w * self.velocity + c1 * r1 * v1 + c2 * r2 * v2

    def update_position(self):
        self.position = self.position + self.velocity
        self.position = np.where(self.position > 0, True, False)
        # Ensure exactly 5 features are selected
        selected_features_count = np.sum(self.position)
        while selected_features_count != 5:
            if selected_features_count > 5:
                # Randomly set some positions to False
                true_indices = np.where(self.position)[0]
                self.position[np.random.choice(true_indices)] = False
            else:
                # Randomly set some positions to True
                false_indices = np.where(~self.position)[0]
                self.position[np.random.choice(false_indices)] = True
            selected_features_count = np.sum(self.position)
def pso_feature_selection_visualized(X, y, n_particles=10, n_iterations=20, model=LinearRegression()):
    num_features = X.shape[1]
    particles = [Particle(num_features) for _ in range(n_particles)]
    global_best_position = np.zeros(num_features, dtype=bool)
    global_best_score = float('inf')
    global_best_scores_history = []  # To record the history of the global best score

    for iteration in range(n_iterations):
        for particle in particles:
            # Select features based on particle's position
            selected_features = particle.position.astype(bool)
            if np.sum(selected_features) == 0:  # Ensure at least one feature is selected
                continue

            X_selected = X.iloc[:, selected_features]
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2)

            # Train the model and calculate the score
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = mean_squared_error(y_test, predictions)

            # Update personal best
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position

            # Update global best
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position

        # Update velocity and position for each particle
        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position()

        global_best_scores_history.append(global_best_score)

    return global_best_position, global_best_scores_history

# Load your dataset
dataset = pd.read_csv("industrial_dataset.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Run PSO for feature selection and get the history of global best scores
_, global_best_scores_history = pso_feature_selection_visualized(X, y)

# Plotting the convergence of the global best score
plt.plot(global_best_scores_history)
plt.xlabel('Iteration')
plt.ylabel('Global Best Score (MSE)')
plt.title('PSO Feature Selection Convergence')
plt.show()