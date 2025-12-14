# -*- coding: utf-8 -*-
# ======================================================
# @File    : HT_Kriging.py
# @Author  : Chen Yuling
# @Date    : 2025/10/15 20:36
# @Desc    : 利用 LightGBM + Kriging 联合预测森林TH空间分布（稳定版）
# ======================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#%% ======================================================
# 1️⃣ 数据读取与预处理
# ======================================================
df1 = pd.read_csv('./DATA/TrainDATA30m.csv', sep=',')
sym90 = pd.read_csv('./DATA/sym90.csv', sep=',')
data = pd.merge(df1, sym90, on='trmc', how='left')

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

# 离散特征处理
for col in ['trzd', 'pnf', 'SU_SYM90']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].round().astype('Int64')
    data[col] = data[col].astype(str)

data = data[data['pnf'] != '0']

# One-Hot 编码
cat_cols = ['trzd', 'SU_SYM90', 'pnf']
data = pd.get_dummies(data, columns=cat_cols,
                      prefix=['TRZD', 'TRMC', 'PNF'],
                      prefix_sep="_", drop_first=False)

predictors = [c for c in data.columns if c not in ['X', 'Y', 'HT', 'REGION']]
X = data[predictors]
y = data['HT']
coords_x = data['X'].values
coords_y = data['Y'].values
w = data['weight'].values if 'weight' in data.columns else np.ones(len(data))

X_train, X_test, y_train, y_test, w_train, w_test, XLAT_train, XLAT_test, YLON_train, YLON_test = train_test_split(
    X, y, w, coords_x, coords_y, test_size=0.33, random_state=2024
)

#%% ======================================================
# 2️⃣ LightGBM 训练
# ======================================================
paras = {
    'reg_alpha': 4.419244901987033,
    'reg_lambda': 0.30434831388967387,
    'num_leaves': 333,
    'min_child_samples': 5,
    'max_depth': 19,
    'learning_rate': 0.05,
    'colsample_bytree': 0.4886694204822875,
    'n_estimators': 7661,
    'n_jobs': -1,
    'random_state': 2024
}
model5 = LGBMRegressor(**paras)
model5.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_test, y_test)], eval_metric='rmse')

pred_train = model5.predict(X_train)
pred_test = model5.predict(X_test)

MSE_train = mean_squared_error(y_train, pred_train)
RMSE_train = np.sqrt(MSE_train)
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

#%% ======================================================
# 3️⃣ LightGBM + Kriging（稳定版）
# ======================================================
import os
import numpy as np
import pandas as pd
import gstools as gs
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1️⃣ 训练集 residual
train_idx = X_train.index
data.loc[train_idx, 'residual'] = y_train - model5.predict(X_train)

# 2️⃣ 可选：训练集 residual 抽样（减少内存）
sample_size = min(20000, len(train_idx))  # 最多 2w 点
sample_idx = np.random.choice(train_idx, size=sample_size, replace=False)
data_train_sample = data.loc[sample_idx]

# 3️⃣ 构建变异函数
var_res = np.var(data_train_sample['residual'])
model = gs.Exponential(dim=2, var=var_res, len_scale=0.1, nugget=0.05*var_res)

# 4️⃣ 分块 + 批量 Kriging
def batch_block_kriging(data_cond, x_pred, y_pred, model, chunk_size=5000, k_neigh=500):
    preds = []
    n = len(x_pred)
    cond_coords = np.column_stack((data_cond['X'], data_cond['Y']))
    cond_values = data_cond['residual'].values
    nbrs = NearestNeighbors(n_neighbors=k_neigh, algorithm='auto').fit(cond_coords)

    for start in tqdm(range(0, n, chunk_size), desc="分块Kriging预测中..."):
        end = min(start + chunk_size, n)
        xi_chunk = x_pred[start:end]
        yi_chunk = y_pred[start:end]

        # 每个 chunk 取 k_neigh 最近邻训练点
        chunk_coords = np.column_stack((xi_chunk, yi_chunk))
        _, idxs = nbrs.kneighbors(chunk_coords)

        # 构建每个chunk的条件点集合（去重）
        unique_idx = np.unique(idxs.flatten())
        sub_coords = cond_coords[unique_idx]
        sub_values = cond_values[unique_idx]

        # 构建 Kriging（一次性预测整个 chunk）
        ok = gs.krige.Ordinary(
            model=model,
            cond_pos=(sub_coords[:,0], sub_coords[:,1]),
            cond_val=sub_values,
            exact=True
        )
        p_chunk, _ = ok((xi_chunk, yi_chunk))
        preds.extend(p_chunk)

    return np.array(preds)

# 5️⃣ 预测测试集 residual
pred_r_test = batch_block_kriging(
    data_cond=data_train_sample,
    x_pred=XLAT_test,
    y_pred=YLON_test,
    model=model,
    chunk_size=5000,
    k_neigh=500
)

# 6️⃣ LGBM 原始预测
y_pred_test_LGBM = model5.predict(X_test)

# 7️⃣ Kriging 校正预测
y_pred_test_Kriging = y_pred_test_LGBM + pred_r_test

# 8️⃣ 性能评估
def evaluate(y_true, y_pred, name="模型"):
    R2 = r2_score(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    MAE = mean_absolute_error(y_true, y_pred)
    print(f"{name}: R²={R2:.4f}, RMSE={RMSE:.4f}, MAE={MAE:.4f}")
    return R2, RMSE, MAE

print("\n=== 测试集性能对比 ===")
evaluate(y_test, y_pred_test_LGBM, name="LGBM 原始预测")
evaluate(y_test, y_pred_test_Kriging, name="LGBM + Kriging 校正预测")