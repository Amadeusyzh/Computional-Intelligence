import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# 加载数据集
data = pd.read_csv('scaled_industrial_dataset_original_index.csv')
X = data.iloc[:, :2]  # 选取前两个特征
y = data.iloc[:, -1]

# 粒子类
class Particle:
    def __init__(self, num_features):
        self.position = np.random.rand(num_features) > 0.5
        self.velocity = np.random.randn(num_features)
        self.best_position = np.copy(self.position)
        self.best_score = -np.inf

    def update_velocity(self, global_best_position, w=0.5, c1=1, c2=2):
        r1, r2 = np.random.rand(2)
        # 将布尔数组转换为整数数组进行计算
        position_int = self.position.astype(int)
        best_position_int = self.best_position.astype(int)
        global_best_position_int = global_best_position.astype(int)
        cognitive_component = c1 * r1 * (best_position_int - position_int)
        social_component = c2 * r2 * (global_best_position_int - position_int)
        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self):
        self.position = np.random.rand(len(self.velocity)) < 1 / (1 + np.exp(-self.velocity))
        self.position = self.position.astype(bool)


# PSO特征选择函数
def pso_feature_selection(X, y, n_particles=10, n_iterations=50, model=LinearRegression()):
    num_features = 2  # 假设总共50个特征
    particles = [Particle(num_features) for _ in range(n_particles)]
    global_best_position = np.zeros(num_features, dtype=bool)
    global_best_score = -np.inf

    positions = []  # 用于存储粒子位置的列表

    for iteration in range(n_iterations):
        for iteration in range(n_iterations):
            for particle in particles:
                selected_features = particle.position
                if not selected_features.any():
                    continue
                X_selected = X.iloc[:, selected_features]
                score = np.mean(cross_val_score(model, X_selected, y, cv=5))
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle.position

        # 记录粒子位置
        positions.append([particle.position for particle in particles])

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position()

    return global_best_position, positions

# 运行PSO特征选择并记录粒子位置
best_features, positions = pso_feature_selection(X, y)

# 创建三维动画
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

def init():
    return fig,

def animate(i):
    ax.clear()
    for p in positions[i]:
# 取前两个特征和粒子位置的第一个维度
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], p[0], marker='o')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Particle Dimension 1')
    return fig,

ani = FuncAnimation(fig, animate, init_func=init, frames=len(positions), interval=200)
ani.save('pso_animation_3d.gif', writer='ffmpeg')