import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 假设您已经有了一个DataFrame 'df'，其中包含特征和目标变量
# X = df.drop('target_column_name', axis=1)
# y = df['target_column_name']
dataset = pd.read_csv("scaled_industrial_dataset.csv")
X = dataset.iloc[:, :-1]  # Assuming the last
y = dataset.iloc[:, -1]

# 计算每个特征与目标变量的相关系数
correlations = X.corrwith(y)

# 选择相关系数最强的N个特征
N = 10 # 可以根据您的需求调整
selected_features = correlations.abs().sort_values(ascending=False).head(N).index

print(selected_features)
# 使用选定的特征进行模型训练
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

# 模型评估
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)
