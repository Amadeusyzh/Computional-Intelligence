import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler




# 定义粒子
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

class Particle:
    def __init__(self, n_features):
        self.position = np.random.rand(n_features) > 0.5  # 随机初始化位置
        self.velocity = np.random.randn(n_features)  # 随机初始化速度
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def update_velocity(self, global_best_position, w, c1, c2):
        r1, r2 = np.random.rand(2, len(self.velocity))
        self.velocity = w * self.velocity + c1 * r1 * (self.best_position - self.position.astype(int)) + c2 * r2 * (global_best_position.astype(int) - self.position.astype(int))

    def update_position(self):
        self.position = np.random.rand(len(self.velocity)) < 1 / (1 + np.exp(-self.velocity))
        self.position = self.position.astype(bool)

def pso_feature_selection(X, y, n_particles=30,
                          n_iterations=100, w=0.4, c1=1.5, c2=1.5,
                          convergence_threshold=0.001,
                          max_convergence_count=1000):
    n_features = X.shape[1]
    particles = [Particle(n_features) for _ in range(n_particles)]
    global_best_position = np.zeros(n_features, dtype=bool)
    global_best_score = float('inf')

    convergence_count = 0
    previous_global_best_score = global_best_score

    for t in range(n_iterations):
        for particle in particles:
            selected_features = particle.position
            if not selected_features.any():
                continue
            X_selected = X.iloc[:, selected_features]
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2)

            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = mean_squared_error(y_test, predictions)

            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = np.copy(particle.position)

            if score < global_best_score:
                global_best_score = score
                global_best_position = np.copy(particle.position)

        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position()

        # 检查收敛条件
        if abs(global_best_score - previous_global_best_score) < convergence_threshold:
            convergence_count += 1
            print(convergence_count)
            if convergence_count >= max_convergence_count:
                break
        else:
            convergence_count = 0
            previous_global_best_score = global_best_score

    return global_best_position
# 加载数据集（您可以替换为您自己的数据集）
# X, y = ...
dataset = pd.read_csv("industrial_dataset.csv",index_col=0)
X = dataset.iloc[:, :-1]  # Assuming the last
y = dataset.iloc[:, -1]

# 使用PSO进行特征选择
selected_features = pso_feature_selection(X, y)
print("Selected Features:", selected_features)
selected_feature_indices = [i+1 for i, selected in enumerate(selected_features) if selected]
print("Indices of Selected Features:", selected_feature_indices)