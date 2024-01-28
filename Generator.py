import pandas as pd
import numpy as np

# 载入原始数据集
file_path = 'industrial_dataset.csv'  # 请替换为您的文件路径
data = pd.read_csv(file_path)

# 去除原始索引列和目标列
features = data.drop(['Unnamed: 0', 'Target'], axis=1)

# 基于原始数据集的统计特性生成新数据点
new_features_based_on_original = pd.DataFrame({col: np.random.normal(loc=features[col].mean(),
                                                                    scale=features[col].std(),
                                                                    size=1000)
                                              for col in features.columns})

# 确保生成的数据在原始数据的范围内
new_features_based_on_original = new_features_based_on_original.clip(lower=features.min(),
                                                                    upper=features.max(),
                                                                    axis=1)

# 如果需要，将生成的数据集保存为CSV文件
new_features_based_on_original.to_csv('generated_dataset.csv', index=False)