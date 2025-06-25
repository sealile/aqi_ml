import pickle

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import shap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（Windows系统自带）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

loc='chengdu'
# locs=['jinan',]#'chengdu','shanghai','tianjin'
file_paths = [f"../2025/combine_{loc}_2025.xlsx"]

# 1. 读取数据
# df = pd.read_excel("./jinan_fuben/combine_2020.xlsx")

dfs = []  # 存储所有DataFrame的列表'
for file in file_paths:
    df = pd.read_excel(file)  # 读取单个文件
    dfs.append(df)            # 将DataFrame添加到列表中

# 合并所有DataFrame（纵向堆叠）
df = pd.concat(dfs, axis=0, ignore_index=True)  # axis=0表示纵向合并


# 2. 数据预处理
X = df.drop(columns=[df.columns[0], df.columns[1]])  # 去除日期和AQI列
y = df.iloc[:, 1]  # 目标列

# 处理缺失值（推荐使用中位数填充避免异常值影响）
X = X.fillna(X.median())
X_test=X
y_test=y


# 6. 训练模型（使用更细致的早停设置）
model = joblib.load(f'{loc}.pkl')
y_pred = model.predict(X_test)
# joblib.dump(model, 'tianjin.pkl')

# 7. 增强版评估
# y_pred = model.predict(X_test)
print("\n=== 综合评估指标 ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")


# 构建包含原始日期、真实AQI、预测AQI 的 DataFrame
result_df = pd.DataFrame({
    'Date': df.iloc[:, 0],          # 第一列：日期
    'Actual_AQI': y_test,           # 第二列：真实 AQI
    'Predicted_AQI': y_pred         # 第三列：预测 AQI
})

# 保存为 Excel 文件
output_path = f'predict_aqi_2025_{loc}.xlsx'
result_df.to_excel(output_path, index=False)

print(f"\n预测结果已保存至：{output_path}")