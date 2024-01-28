import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取数据集
data = pd.read_csv("industrial_dataset.csv")

# 保存原始索引和目标变量
original_index = data.iloc[:, 0]  # 假设第一列是索引
target = data.iloc[:, -1]         # 假设最后一列是目标变量

# 分离特征
X = data.iloc[:, 1:-1]  # 特征，除去第一列（索引）和最后一列（目标变量）

# 初始化 StandardScaler
scaler = StandardScaler()

# 对特征进行拟合和转换
scaled_features = scaler.fit_transform(X)

# 创建一个新的DataFrame，包含了标准化的特征
scaled_data = pd.DataFrame(scaled_features, columns=X.columns)

# 将原始索引和目标变量添加回数据集
scaled_data.insert(0, 'Original_Index', original_index)
scaled_data['Target'] = target

# 将处理后的数据保存到文件
scaled_data.to_csv('scaled_industrial_dataset.csv', index=False)
