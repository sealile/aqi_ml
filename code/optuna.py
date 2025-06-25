from math import sqrt

import pip
import plotly.io as pio
import optuna
import pandas as pd
import numpy as np
from optuna.visualization import plot_param_importances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import shap
import kaleido


# loc='jinan'
locs=['chengdu']  #,'chengdu','shanghai','tianjin'
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



def objective(trial ,data=X ,target=y):

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2 ,random_state=42)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": 'gbdt',
        "learning_rate": 0.1,
        'n_estimators': 1000,
        "verbose": -1,

        "num_leaves": trial.suggest_int("num_leaves", 30,100),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "subsample": trial.suggest_categorical("subsample", [0.8,0.85,0.9,0.85,1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 0.85, 0.9, 0.95, 1.0]),
        "lambda_l1": trial.suggest_categorical("lambda_l1", [0,1,2,3,4,5,6,7,8,9,10]),
        "lambda_l2": trial.suggest_categorical("lambda_l2", [0,1,2,3,4,5,6,7,8,9,10]),

    }
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="mae",  # 增大早停耐心值
        callbacks=[
            # lgb.log_evaluation(period=50),  # 每50轮输出日志
            lgb.early_stopping(stopping_rounds=100),
        ]
    )

    preds = model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, preds))

    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print('------------------------------')
print(f"\tBest value (rmse): {study.best_value:.5f}")
print(f"\tBest params:")
for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")


optuna.visualization.plot_optimization_history(study).show()  # 显示优化历史

# 获取所有试验数据并转为 DataFrame
trials_df = study.trials_dataframe()

# 保存到 Excel
trials_df.to_excel("optimization_history.xlsx", index=False)

 #获取每次试验后的最佳值
history = []
best_value = float('inf')  # 假设是最小化问题
for trial in study.trials:
    if trial.value is not None:  # 只考虑成功的试验
        if trial.value < best_value:  # 更新最佳值
            best_value = trial.value
        history.append({
            "trial_number": trial.number,
            "best_value": best_value,
        })

# 转为 DataFrame 并保存
history_df = pd.DataFrame(history)
history_df.to_excel("best_values_history.xlsx", index=False)


# optuna.visualization.plot_param_importances(study).show()     # 显示参数重要性


import plotly.io as pio
# print(pio.kaleido.scope.state.executable_path)

# 生成参数重要性图表对象
# fig = plot_param_importances(study)

# pio.write_image(fig, "param_importance.svg", format="svg")

