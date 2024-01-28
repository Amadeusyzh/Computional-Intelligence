import pandas as pd
from joblib import load
from icecream import ic
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 加载之前保存的模型
linear_regression_model = load('linear_regression_model.joblib')
svm_model = load('svm_model.joblib')

dataset = pd.read_csv("generated_dataset.csv")

selected_columns = dataset.columns[[1, 3, 4, 7, 42]]  # Adjust indices for zero-based indexing
X_selected = dataset[selected_columns]
print(X_selected)

# 使用加载的模型进行预测
# 假设 new_data 是您要预测的新数据集
predictions_lr = linear_regression_model.predict(X_selected)
predictions_svm = svm_model.predict(X_selected)
ic(predictions_lr)
ic(predictions_svm)


df_predictions_lr = pd.DataFrame(predictions_lr, columns=['Linear_Regression_Predictions'])
df_predictions_svm = pd.DataFrame(predictions_lr, columns=['SVM_Predictions'])
# 将DataFrame保存为CSV文件
df_predictions_lr.to_csv('predictions_lr.csv', index=False)
df_predictions_lr.to_csv('predictions_svm.csv', index=False)



