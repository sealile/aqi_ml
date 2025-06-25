import pandas as pd
import numpy as np
from pyexpat import features
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import shap


plt.rcParams.update({
    "font.family": "serif",        # 使用衬线字体（Times New Roman属于衬线字体）
    "font.serif": ["Times New Roman"],  # 指定衬线字体为Times New Roman
    "font.size": 16,               # 全局字号（标题、标签等可单独调整）
    "axes.titlesize": 18,          # 标题字号

})
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# loc='jinan'
locs=['jinan']
file_paths = [f"./{loc}/combine_{loc}_{year}.xlsx" for loc in locs for year in range(2020, 2025)]

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


# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# # 4. 标准化处理（注意：仅在训练集上拟合scaler）
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)  # 测试集使用相同的scaler
#
# # 转换为DataFrame保持列名（可选）
# X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
# X_test = pd.DataFrame(X_test_scaled, columns=X.columns)


# 5. 配置增强版参数
params = {
    "objective": "regression",
    "metric": ["rmse"],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    'max_depth': 9,
    # 适当增加复杂度
    "learning_rate": 0.1,
    "n_estimators": 1000,        # 增加树的数量
    "lambda_l1": 5,            # L1正则化
    "lambda_l2": 0,           # L2正则化
    "subsample": 0.85,     # 特征采样比例
    "colsample_bytree": 0.8,           # 每5次迭代执行bagging
    "verbose": -1
}

# 6. 训练模型（使用更细致的早停设置）
model = lgb.LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse", # 增大早停耐心值
    callbacks=[
        # lgb.log_evaluation(period=10),  # 每50轮输出日志
        lgb.early_stopping(stopping_rounds=100),
        # lgb.reset_parameter(learning_rate=lambda epoch: 0.03 * (0.99 ** epoch))  # 动态学习率衰减
    ]
)

# 使用SHAP解释模型
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# feature_names = X.columns.tolist()
feature_names=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                'AVG_T', 'AVG_DP', 'MAX_T', 'MIN_T', 'AVG_V', 'AVG_WS', 'MAX_WS', 'PR', 'RH']

# 全局特征重要性（条形图）
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values,max_display=15,show=False)
plt.savefig("tiaoxing.svg", format="svg", bbox_inches="tight")
plt.close()

# 特征影响分布图（蜂群图）
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_names,show=False)
plt.savefig("fengqun.svg", format="svg", bbox_inches="tight")
plt.close()




# 7. 增强版评估
y_pred = model.predict(X_test)
print("\n=== 综合评估指标 ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

