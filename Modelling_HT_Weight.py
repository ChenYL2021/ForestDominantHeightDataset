# -*- coding: utf-8 -*-
# @Time    : 2025/10/15 10:52
# @Author  : CYL
# @File    : HT_Weight.py
# @Describe: LGBM矫正（权重，生态区权重） ：重新调参


#%%忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,r2_score

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt


#%% ======================================================
# 1 数据处理与划分
# ========================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 数据读取与基础合并

# 读取主样点与生态分区编码表
df1 = pd.read_csv('./DATA/TrainDATA30m.csv', sep=',')
sym90 = pd.read_csv('./DATA/sym90.csv', sep=',')  # trmc代码转名称

# 合并数据（左连接，保留主数据完整性）
data = pd.merge(df1, sym90, on='trmc', how='left')

#保留关键字段并去重
keep_cols = [
    'X', 'Y', 'HT', 'REGION',
    'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5',
    'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
    'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15',
    'bio_16', 'bio_17', 'bio_18', 'bio_19',
    'age', 'VH', 'VV', 'tchd', 'trzd', 'aspect',
    'elevation', 'slope', 'pnf', 'NDVI_MAX', 'SU_SYM90'
]
data = data[keep_cols].drop_duplicates().dropna(subset=['HT', 'X', 'Y'])
#离散特征预处理（避免数值与类别混用）

# 将地类/坡向/主类编码等转为字符串型

for col in ['trzd', 'pnf', 'SU_SYM90']:
    # 转换为 float，忽略无法转换的值
    data[col] = pd.to_numeric(data[col], errors='coerce')
    # 四舍五入并转为整数类型
    data[col] = data[col].round().astype('Int64')
    # 转为字符串
    data[col] = data[col].astype(str)
# 可选：过滤无效样点（如森林覆盖度 pnf=0）
data = data[data['pnf'] != '0']
# One-Hot 编码
cat_cols = ['trzd', 'SU_SYM90', 'pnf']
data = pd.get_dummies(
    data,
    columns=cat_cols,
    prefix=['TRZD', 'TRMC', 'PNF'],
    prefix_sep="_",
    dummy_na=False,
    drop_first=False
)

#%% ======================================================
# 2 Declustering 权重计算(生态区权重）
# ========================================================
from tqdm import tqdm
import pandas as pd
import numpy as np
def compute_declustering_weights_balanced(df, x_col='X', y_col='Y', grid_size=0.01, alpha=0.5):
    """
    平滑 Declustering 权重（兼顾稳定性和稀疏差异）
    ------------------------------------------------
    参数：
        grid_size : 网格分辨率
        alpha     : 平滑指数（0~1），控制权重平滑程度
                    - 较小（如 0.3） → 更平衡
                    - 较大（如 0.7） → 稀疏权重更突出
    """
    df = df.copy()

    # 计算网格
    df['grid_x'] = np.floor(df[x_col] / grid_size).astype(int)
    df['grid_y'] = np.floor(df[y_col] / grid_size).astype(int)
    df['grid_id'] = df['grid_x'].astype(str) + "_" + df['grid_y'].astype(str)

    # 每个网格的样点数
    grid_counts = df.groupby('grid_id')[x_col].transform('count')

    # 平滑反比权重：使用 log 缓解极端差异
    df['weight'] = 1 / np.power(np.log1p(grid_counts), alpha)

    # 归一化到 1
    df['weight'] /= df['weight'].sum()

    print(f"✅ 平滑 Declustering 权重计算完成 (α={alpha})")
    print(f"范围: {df['weight'].min():.6e} ~ {df['weight'].max():.6e}")
    print(f"总和: {df['weight'].sum():.6f}")

    return df['weight']

# 使用示例
data['weight'] = compute_declustering_weights_balanced(
    data, x_col='X', y_col='Y', grid_size=0.01, alpha=0.5
)

print("✅ Declustering 权重计算完成")
print(f"权重范围: {data['weight'].min():.6f} ~ {data['weight'].max():.6f}")
print(f"权重总和: {data['weight'].sum():.6f}")
print(data[['REGION', 'weight']].groupby('REGION').sum().head())


#%% 生成训练特征与目标
# 自动获取所有哑变量后的特征列

predictors = [c for c in data.columns if c not in ['X', 'Y', 'HT', 'REGION']]
X = data[predictors]
y = data['HT']

#数据划分（空间坐标同步划分）
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, data['weight'], test_size=0.33, random_state=2024
)
#%% ======================================================
# 3 LGBM + Optuna
# ========================================================
import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from optuna.visualization import plot_optimization_history, plot_param_importances

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 11, 333),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01, 0.02, 0.05, 0.1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 7000, 8000),
        'n_jobs': -1,
        'force_col_wise': True,
        'random_state': 2024,
    }

    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[]
    )

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

# 创建 study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 输出结果
print("✅ Best RMSE:", study.best_value)
print("✅ Best params:", study.best_params)

#%% 可视化结果
import plotly.io as pio
pio.renderers.default = 'browser'
plot_optimization_history(study).show()
plot_param_importances(study).show()

#%%
from lightgbm import LGBMRegressor
paras = {'reg_alpha': 0.004799331089056598, 'reg_lambda': 1.3887180052747095, 'num_leaves': 193, 'min_child_samples': 65, 'max_depth': 14, 'learning_rate': 0.1, 'colsample_bytree': 0.3125653773611078,
          'n_estimators': 7949, 'random_state': 2024}
model5A = LGBMRegressor(**paras)
model5A.fit(X_train, y_train, sample_weight=w_train)


#%% 计算指标
pred_train = model5A.predict(X_train)
pred_test  = model5A.predict(X_test)

# mean_squared_error() 在老版 sklearn 中返回 MSE（没有 squared 参数）
MSE_train = mean_squared_error(y_train, pred_train)      # MSE
RMSE_train = np.sqrt(MSE_train)                          # RMSE
MAE_train = mean_absolute_error(y_train, pred_train)
R2_train = r2_score(y_train, pred_train)

MSE_test = mean_squared_error(y_test, pred_test)
RMSE_test = np.sqrt(MSE_test)
MAE_test = mean_absolute_error(y_test, pred_test)
R2_test = r2_score(y_test, pred_test)

print("=== 训练集性能指标 ===")
print(f"R2_train  = {R2_train:.4f}")
print(f"RMSE_train = {RMSE_train:.4f}")
print(f"MSE_train  = {MSE_train:.4f}")
print(f"MAE_train  = {MAE_train:.4f}")

print("\n=== 测试集性能指标 ===")
print(f"R2_test  = {R2_test:.4f}")
print(f"RMSE_test = {RMSE_test:.4f}")
print(f"MSE_test  = {MSE_test:.4f}")
print(f"MAE_test  = {MAE_test:.4f}")

