import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# 加载数据集
# 确保替换为您的数据集路径
data = pd.read_csv('scaled_industrial_dataset_original_index.csv', index_col=0)  # 假设第一列是索引
X = data.iloc[:, :-1]  # 选取除了最后一列之外的所有列作为特征
y = data.iloc[:, -1]   # 最后一列是目标变量


# 粒子类
class Particle:
    def __init__(self, n_features):
        self.position = np.random.rand(n_features) > 0.5  # 随机初始化位置
        self.velocity = np.random.randn(n_features)  # 随机初始化速度
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def update_velocity(self, global_best_position, w, c1, c2):
        r1, r2 = np.random.rand(2, len(self.velocity))

        # 将布尔数组转换为整数数组
        position_int = self.position.astype(int)
        best_position_int = self.best_position.astype(int)
        global_best_position_int = global_best_position.astype(int)

        # 执行速度更新
        self.velocity = w * self.velocity + c1 * r1 * (best_position_int - position_int) + c2 * r2 * (
                    global_best_position_int - position_int)
    def update_position(self):
        self.position = np.random.rand(len(self.velocity)) < 1 / (1 + np.exp(-self.velocity))
        self.position = self.position.astype(bool)

# PSO特征选择函数
def pso_feature_selection(X, y, n_particles=30, n_iterations=1000, w_start=0.9,
                          w_end=0.4, c1_start=2.5, c1_end=0.5, c2_start=0.5, c2_end=2.5,
                          convergence_threshold=0.001, max_convergence_count=1000):
    n_features = X.shape[1]
    particles = [Particle(n_features) for _ in range(n_particles)]
    global_best_position = np.zeros(n_features, dtype=bool)
    global_best_score = float('inf')

    convergence_count = 0
    previous_global_best_score = global_best_score

    for t in range(n_iterations):
        w = w_end + (w_start - w_end) * (1 - t / n_iterations)
        c1 = c1_end + (c1_start - c1_end) * (1 - t / n_iterations)
        c2 = c2_end + (c2_start - c2_end) * (1 - t / n_iterations)

        for particle in particles:
            # 评估粒子适应度
            selected_features = particle.position
            if not selected_features.any():
                continue
            X_selected = X.iloc[:, selected_features]
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2)

            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = mean_squared_error(y_test, predictions)

            # 更新粒子的个人最佳位置
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = np.copy(particle.position)

            # 更新全局最佳位置
            if score < global_best_score:
                global_best_score = score
                global_best_position = np.copy(particle.position)

        for particle in particles:
            # 更新粒子的速度和位置
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position()

            # 检查收敛条件
        if abs(global_best_score - previous_global_best_score) < convergence_threshold:
            convergence_count += 1

            if convergence_count >= max_convergence_count:
                print("count:",convergence_count)
                print("w:",w, "c1:",c1, "c1:",c2)
                break
        else:
            print(convergence_count)
            print("w:",w, "c1:",c1, "c1:",c2)
            convergence_count = 0
            previous_global_best_score = global_best_score

    return global_best_position
# 使用PSO选择特征
best_features = pso_feature_selection(X, y)# 假设 best_features 是选中特征的布尔数组

print("Selected Features:", best_features)
selected_feature_indices = [i+1 for i, selected in enumerate(best_features) if selected]
print("Indices of Selected Features:", selected_feature_indices)