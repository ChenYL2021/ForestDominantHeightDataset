# -*- coding: utf-8 -*-
# @Time    : 2025/10/1 21:42
# @Author  : ChenYuling
# @Software: PyCharm
# @Describe：对比两种方式进行优势高模型构建

#%%忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#%read data
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt
import pandas as pd
import numpy as np
import lightgbm as lgb
import gstools as gs
import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#%%#####################################################################################################################
###########################################读取数据进行哑变量化处理#################################################
########################################################################################################################
#%read data
# df1 = pd.read_csv('./DATA/TrainDATA30m.csv', sep=',')
df1 = pd.read_csv(r'K:\WorkingNotes\TH\WN\20251005\DATA\data2.csv',encoding = 'gb2312') #,encoding = 'gb2312'
#%
sym90 = pd.read_csv('./DATA/sym90.csv', sep=',')#trmc代码转名称, encoding = 'gb2312'
data1 = pd.merge(df1, sym90, on='trmc', how='left')
#%
data2 = data1[['X', 'Y','HT',
        'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
        'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age',
        'VH', 'VV', 'tchd', 'trzd', 'aspect', 'elevation', 'slope', 'pnf', 'NDVI_MAX', 'SU_SYM90_y','REGION']]
data2 = data2.dropna(axis=0,how='any')
data2_filtered = data2[data2['pnf'] != 0]
#%处理离散数据-哑变量处理
# 将列转换为字符类型
# data2['trzd'] = data2['trzd'].round().astype('Int64')
data2_filtered['trzd'] = data2_filtered['trzd'].astype(str)
# data2['pnf'] = data2['pnf'].round().astype('Int64')
data2_filtered['pnf'] = data2_filtered['pnf'].astype(str)
# sub_df['pnf'] = round(sub_df['pnf'])
data3 = pd.get_dummies(
    data2_filtered,
    columns=['trzd','SU_SYM90_y','pnf'],
    prefix=['TRZD','TRMC','PNF'],
    prefix_sep="_",
    dummy_na=False,
    drop_first=False)
data3.columns
#%%#####################################################################################################################
"""
全国森林样地建模脚本（HT为目标变量）
流程：
1. Declustering（1km Grid-based）
2. LightGBM建模
3. 残差Coarse-grid Kriging
4. 输出预测+不确定度
"""


#% =========================
# 1. Declustering: 生态区 + 空间网格权重
# =========================
def compute_ecogrid_weights(df, x_col='X', y_col='Y', eco_col='REGION', grid_size=1000, all_ecos=None):
    weights = pd.Series(0.0, index=df.index)
    if all_ecos is None:
        all_ecos = df[eco_col].unique()

    for eco in all_ecos:
        eco_df = df[df[eco_col] == eco].copy()
        if eco_df.empty:
            continue

        # 空间网格索引
        eco_df['grid_x'] = (eco_df[x_col] // grid_size).astype(int)
        eco_df['grid_y'] = (eco_df[y_col] // grid_size).astype(int)
        eco_df['grid_id'] = eco_df['grid_x'].astype(str) + "_" + eco_df['grid_y'].astype(str)

        # 网格内权重 = 1 / 网格点数
        grid_counts = eco_df.groupby('grid_id').size()
        eco_df['weight'] = eco_df['grid_id'].map(lambda g: 1.0 / grid_counts[g])

        # 归一化，使该生态区总权重 = 1
        eco_df['weight'] = eco_df['weight'] / eco_df['weight'].sum()
        weights.loc[eco_df.index] = eco_df['weight']

    return weights


# 计算权重
all_eco_ids = data3['REGION'].unique()
weights = compute_ecogrid_weights(data3, grid_size=1000, all_ecos=all_eco_ids)
data3['weight'] = weights

#%% ============ 3. LightGBM建模 ============功能: 使用 Declustering 权重 + 自动调参 + 早停 + 可视化
# ==========================================================
# 3.1. 数据准备
# ==========================================================
# 因变量与自变量
y = data3['HT'].astype(float).values
X = data3.drop(columns=['HT', 'X', 'Y','REGION']).copy()
w = data3['weight'].astype(float).values

# 防止内存不足：LightGBM 能原生支持 float32
X = X.astype(np.float32)

# 划分训练集与验证集
X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(
    X, y, w, test_size=1/3, random_state=2025
)

train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, weight=w_valid)

#%% ==========================================================
# 3.2 Optuna + LightGBM 调参
# ==========================================================
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 120),
        'max_depth': trial.suggest_int('max_depth', 5, 16),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 1000),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'seed': 2025,
        'feature_pre_filter': False  # 可留也可删，已无冲突
    }

    # ⚠️ 每次 trial 内部重建 Dataset，防止参数冲突
    train_data_trial = lgb.Dataset(X_train, label=y_train)
    valid_data_trial = lgb.Dataset(X_valid, label=y_valid)

    # 训练模型
    gbm = lgb.train(
        params,
        train_data_trial,
        num_boost_round=3000,
        valid_sets=[valid_data_trial],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )

    # 预测与评价
    preds = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse

# =======================
# 3️.3  运行 Optuna 优化
# =======================
print("\n⏳ 开始 Optuna 自动调参 ...")
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=2025))
study.optimize(objective, n_trials=100, show_progress_bar=True)


#%% ==========================================================
# 3.4. 输出最优结果
# ==========================================================
print("\n✅ 最优 RMSE:", study.best_value)
print("✅ 最优参数：")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

#%% ==========================================================
# 3.5. 用最优参数训练最终模型
# ==========================================================
best_params = study.best_params.copy()
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1,
})

# 尝试GPU加速
try:
    best_params["device_type"] = "gpu"
    print("⚙️ 使用 GPU 加速训练")
except Exception:
    pass

train_data = lgb.Dataset(X, label=y, weight=w)

print(f"🔍 当前 LightGBM 版本: {lgb.__version__}")
print("📘 使用 callback 实现早停 (全版本兼容)")

# ✅ 使用 callback 控制早停与日志
callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=200)
]

final_model = lgb.train(
    params=best_params,
    train_set=train_data,
    num_boost_round=2000,
    valid_sets=[train_data],
    valid_names=["train"],
    callbacks=callbacks
)

# final_model.save_model("LGBM_Optuna_ChinaForestModel.txt")
# print("\n📦 模型已保存：LGBM_Optuna_ChinaForestModel.txt")
#%% ==========================================================
# 3.6. 保存模型与 Optuna 结果
# ==========================================================
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import os
# 自动保存路径
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"LGBM_Optuna_ChinaForestModel_{timestamp}.txt"
study_csv = f"Optuna_Study_{timestamp}.csv"
study_pkl = f"Optuna_Study_{timestamp}.pkl"

# 确保当前目录下保存
final_model.save_model(model_path)
joblib.dump(study, study_pkl)

# Optuna trials 数据保存为 CSV
df_trials = study.trials_dataframe()
df_trials.to_csv(study_csv, index=False)

print(f"\n📦 模型已保存: {model_path}")
print(f"📊 Optuna结果已保存为：\n- CSV: {study_csv}\n- PKL: {study_pkl}")

# ✅ 验证保存效果
print(f"\n最优RMSE: {study.best_value:.4f}")
print("最优参数：")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

#%% ==========================================================
# 7. 可视化结果
# ==========================================================
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.io as pio
import optuna
import gstools as gs
try:
    import optuna.visualization as vis

    # RMSE 优化历史
    fig1 = vis.plot_optimization_history(study)
    fig1.update_layout(title="Optuna RMSE Optimization History")

    # 参数重要性
    fig2 = vis.plot_param_importances(study)
    fig2.update_layout(title="Optuna Parameter Importance")

    # 显示交互式图
    if show_browser:
        fig1.show(renderer="browser")
        fig2.show(renderer="browser")

    # 保存静态图
    try:
        rmse_path = os.path.join(vis_dir, f"RMSE_history_{timestamp}.png")
        param_path = os.path.join(vis_dir, f"Param_importance_{timestamp}.png")
        pio.write_image(fig1, rmse_path, scale=3)
        pio.write_image(fig2, param_path, scale=3)
        print(f"✅ 可视化图已保存：\n- {rmse_path}\n- {param_path}")
    except Exception:
        print("⚠️ PNG 导出失败，请确认 kaleido 已安装：pip install -U kaleido")
except Exception as e:
    print("⚠️ Optuna 可视化不可用，请安装：pip install optuna[visualization] plotly")
    print("详细错误：", e)

#%% ============ 4. Coarse-grid Kriging（残差） ============
print("Running coarse-grid kriging on residuals...")

# 聚合残差到 coarse grid (1 km)
coarse_size = 1000
data3['cx'] = (data3['X'] // coarse_size) * coarse_size + coarse_size / 2
data3['cy'] = (data3['Y'] // coarse_size) * coarse_size + coarse_size / 2

coarse_df = data3.groupby(['cx', 'cy']).agg({'residual':'mean'}).reset_index()
cx, cy, cz = coarse_df['cx'].values, coarse_df['cy'].values, coarse_df['residual'].values

# 拟合变差函数
model_vario = gs.Exponential(dim=2)
fit_vario = gs.vario_estimate_unstructured((cx, cy), cz, bin_num=20)
model_vario.fit_variogram((fit_vario[0], fit_vario[1]), nugget=True)

# 构建kriging对象
ok = gs.krige.Ordinary(model_vario, cond_pos=(cx, cy), cond_val=cz)

# 在样点位置预测残差修正量
pred_r, var_r = ok((data3['X'].values, data3['Y'].values))
data3['kriged_residual'] = pred_r
data3['kriged_var'] = var_r

# ============ 5. 合成最终预测与不确定度 ============
data3['HT_final'] = data3['pred_lgbm'] + data3['kriged_residual']
data3['HT_sd'] = np.sqrt(data3['kriged_var'])

# ============ 6. 保存结果 ============
out_cols = ['X', 'Y', 'HT', 'HT_final', 'HT_sd', 'pred_lgbm', 'kriged_residual']
data3[out_cols].to_csv("HT_model_results.csv", index=False)
print("✅ 完成建模！结果已保存为 HT_model_results.csv")

# ============ 7. 可选：输出模型特征重要性 ============
import matplotlib.pyplot as plt
lgb.plot_importance(model, max_num_features=20, figsize=(8,6))
plt.tight_layout()
plt.savefig("feature_importance.jpg", dpi=600)
plt.close()
print("Feature importance plot saved.")

#%%#####################################################################################################################
###########################################独立检验数据集汇总结果（未调参数）#################################################
########################################################################################################################
#%% ==========================================================
# 全国森林样地建模流程封装函数
#=============================================================
#%% ==========================================================
# 全国森林样地建模流程（连续脚本版）
# 1️⃣ Declustering + 2️⃣ LightGBM + Optuna 调参 + 3️⃣ Kriging
#=============================================================
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import gstools as gs
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime
import joblib
#%%read data
import pandas as pd

# 1读取数据
# df1 = pd.read_csv('./DATA/TrainDATA30m.csv', sep=',')
df1 = pd.read_csv(r'K:\WorkingNotes\TH\WN\20251005\DATA\data2.csv', encoding='gb2312')

sym90 = pd.read_csv('./DATA/sym90.csv', sep=',')  # trmc代码转名称

#  合并数据
data1 = pd.merge(df1, sym90, on='trmc', how='left')

# 筛选所需列 & 删除缺失
cols = ['X', 'Y','HT','REGION',
        'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
        'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age',
        'VH', 'VV', 'tchd', 'trzd', 'aspect', 'elevation', 'slope', 'pnf', 'NDVI_MAX', 'SU_SYM90_y']
data2 = data1[cols].dropna(axis=0, how='any')

# 过滤 pnf 非零
data2_filtered = data2.loc[data2['pnf'] != 0].copy()

# 处理离散列-哑变量
data2_filtered.loc[:, 'trzd'] = data2_filtered['trzd'].astype(str)
data2_filtered.loc[:, 'pnf'] = data2_filtered['pnf'].astype(str)

# 哑变量处理
data3 = pd.get_dummies(
    data2_filtered,
    columns=['trzd','SU_SYM90_y','pnf'],
    prefix=['TRZD','TRMC','PNF'],
    prefix_sep="_",
    dummy_na=False,
    drop_first=False
)

# 查看列名
print(data3.columns)

#%% ===============================
# 配置参数
# ===============================
target = 'HT'
x_col, y_col = 'X', 'Y'
eco_col = 'REGION'
grid_size = 1000
coarse_size = 0.01 #0.01 度（约 1 km）
test_size = 1/3
n_trials = 50
vis_dir = "Optuna_Figures"
save_dir = "ChinaForestModel"
show_browser = True
seed = 2025

os.makedirs(vis_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#%% ===============================
# 1️⃣ Declustering 权重计算
# ===============================
def compute_ecogrid_weights(df, x_col, y_col, eco_col, grid_size):
    weights = pd.Series(0.0, index=df.index)
    all_ecos = df[eco_col].unique()

    for eco in all_ecos:
        eco_df = df[df[eco_col] == eco].copy()
        if eco_df.empty:
            continue

        eco_df['grid_x'] = (eco_df[x_col] // grid_size).astype(int)
        eco_df['grid_y'] = (eco_df[y_col] // grid_size).astype(int)
        eco_df['grid_id'] = eco_df['grid_x'].astype(str) + "_" + eco_df['grid_y'].astype(str)

        grid_counts = eco_df.groupby('grid_id').size()
        eco_df['weight'] = eco_df['grid_id'].map(lambda g: 1.0 / grid_counts[g])
        eco_df['weight'] = eco_df['weight'] / eco_df['weight'].sum()
        weights.loc[eco_df.index] = eco_df['weight']

    return weights

data3['weight'] = compute_ecogrid_weights(data3, x_col, y_col, eco_col, grid_size)

#%% ===============================
# 2️⃣ LightGBM 建模 + Optuna 调参
# ===============================
y = data3[target].astype(float).values
X = data3.drop(columns=[target, x_col, y_col, eco_col]).copy()
w = data3['weight'].astype(float).values
X = X.astype(np.float32)

# 划分训练集
X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(
    X, y, w, test_size=test_size, random_state=seed
)

# Optuna 目标函数
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 120),
        'max_depth': trial.suggest_int('max_depth', 5, 16),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 1000),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'seed': seed,
    }

    train_data_trial = lgb.Dataset(X_train, label=y_train)
    valid_data_trial = lgb.Dataset(X_valid, label=y_valid)

    gbm = lgb.train(
        params,
        train_data_trial,
        num_boost_round=3000,
        valid_sets=[valid_data_trial],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )

    preds = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse

# Optuna 调参
print("\n⏳ 开始 Optuna 自动调参 ...")
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
print(f"\n✅ 最优 RMSE: {study.best_value:.4f}")

#%% 使用最优参数训练最终模型
best_params = study.best_params.copy()
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1,
})
try:
    best_params['device_type'] = 'gpu'
    print("⚙️ 使用 GPU 加速训练")
except:
    print("CPU运行！")
    pass

train_data_full = lgb.Dataset(X, label=y, weight=w)
callbacks = [lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=200)]
final_model = lgb.train(
    params=best_params,
    train_set=train_data_full,
    num_boost_round=2000,
    valid_sets=[train_data_full],
    valid_names=["train"],
    callbacks=callbacks
)

# 保存模型和 Optuna study
model_path = os.path.join(save_dir, f"LGBM_Optuna_ChinaForestModel_{timestamp}.txt")
final_model.save_model(model_path)
study_pkl = os.path.join(save_dir, f"Optuna_Study_{timestamp}.pkl")
joblib.dump(study, study_pkl)
print(f"📦 模型已保存: {model_path}")
print(f"📊 Optuna study 已保存: {study_pkl}")

#%% ===============================
# 3️⃣ 可视化 Optuna 结果
# ===============================
try:
    import optuna.visualization as vis
    fig1 = vis.plot_optimization_history(study)
    fig1.update_layout(title="Optuna RMSE Optimization History")
    fig2 = vis.plot_param_importances(study)
    fig2.update_layout(title="Optuna Parameter Importance")

    if show_browser:
        fig1.show(renderer="browser")
        fig2.show(renderer="browser")

    rmse_path = os.path.join(vis_dir, f"RMSE_history_{timestamp}.png")
    param_path = os.path.join(vis_dir, f"Param_importance_{timestamp}.png")
    try:
        pio.write_image(fig1, rmse_path, scale=3)
        pio.write_image(fig2, param_path, scale=3)
        print(f"✅ 可视化图已保存：\n- {rmse_path}\n- {param_path}")
    except:
        print("⚠️ PNG 导出失败，请安装 kaleido: pip install -U kaleido")
except Exception as e:
    print("⚠️ Optuna 可视化不可用，请安装 optuna[visualization] plotly")
    print(e)

#%% ===============================
# 4️⃣ 计算残差 & Coarse-grid Kriging
# ===============================
preds_full = final_model.predict(X)
data3['residual'] = y - preds_full

# 设置 coarse grid
data3['cx'] = (data3[x_col] // coarse_size) * coarse_size + coarse_size / 2
data3['cy'] = (data3[y_col] // coarse_size) * coarse_size + coarse_size / 2

# 聚合计算 coarse 残差
coarse_df = data3.groupby(['cx', 'cy']).agg({'residual': 'mean'}).reset_index()
cx, cy, cz = coarse_df['cx'].values, coarse_df['cy'].values, coarse_df['residual'].values

# 计算经验变异函数
# bin_center, gamma = gs.vario_estimate_unstructured((cx, cy), cz)
# 设置最大距离为样点间最大距离的一半，分10~15个bin较合适
max_dist = np.max(np.sqrt((cx[:, None] - cx[None, :])**2 + (cy[:, None] - cy[None, :])**2)) / 2
bin_edges = np.linspace(0, max_dist, 16)  # 15个bin -> 16个边界
bin_center, gamma = gs.vario_estimate_unstructured(
    (cx, cy), cz,
    bin_edges=bin_edges
)

# 去除nan值
mask = ~np.isnan(gamma)
bin_center, gamma = bin_center[mask], gamma[mask]
#定义理论变差模型
model_vario = gs.Exponential(dim=2)
#拟合变差函数
model_vario.fit_variogram(bin_center, gamma, nugget=True)
#普通克里金插值
ok = gs.krige.Ordinary(model_vario, cond_pos=(cx, cy), cond_val=cz)
pred_r, var_r = ok((data3[x_col].values, data3[y_col].values))

# 保存结果
data3['kriged_residual'] = pred_r
data3['kriged_var'] = var_r
data3['pred_final'] = preds_full + pred_r

print("✅ 全国森林样地建模完成，结果包含 ['weight','residual','kriged_residual','kriged_var','pred_final']")
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(bin_center, gamma, label="Empirical Variogram", color="black", s=35)
plt.plot(bin_center, model_vario.variogram(bin_center),
         label="Fitted Exponential Model", color="red", lw=2)
plt.xlabel("Lag distance (m)", fontsize=11)
plt.ylabel("Semivariance γ(h)", fontsize=11)
plt.title("Residual Variogram Fitting", fontsize=12)
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
data3.to_csv(r"./ChinaForestModel/data3.csv", index=False, encoding="utf-8-sig")