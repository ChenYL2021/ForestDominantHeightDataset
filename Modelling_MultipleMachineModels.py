# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 8:42
# @Author  : ChenYuling
# @Software: PyCharm
# @Describe：将chm和lidar数据进行汇总，然后进行模型训练初探

#%%忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")
# import matplotlib.pyplot as plt
#%read data
# import seaborn as sns
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
from sklearn.model_selection import train_test_split, KFold
import joblib
#%%#####################################################################################################################
###########################################读取数据进行哑变量化处理#################################################
########################################################################################################################
#%read data
df1 = pd.read_csv(r'E:\Biomass\hData\HT\htDATA.csv', sep=',')


#%%
import pandas as pd
import matplotlib.pyplot as plt

# 假设 df1 是你的数据框
# df1 = pd.DataFrame({'Biomass': [...加载你的数据...]})

# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(df1['Ht'], bins=20, edgecolor='black', color='skyblue')  # bins 决定分箱数
plt.title('Frequency Distribution of Biomass', fontsize=16)
plt.xlabel('Ht', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%% 计算四分位数和四分位距
Q1 = df1['Ht'].quantile(0.25)
Q3 = df1['Ht'].quantile(0.75)
IQR = Q3 - Q1
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
# 定义异常值的上下界
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
#% 去除异常值
df2 = df1[(df1['Ht'] >= lower_bound) & (df1['Ht'] <= upper_bound)]
print(f"原始数据大小: {df1.shape}")
print(f"去除异常值后的数据大小: {df2.shape}")

#%% 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(df2['Ht'], bins=20, edgecolor='black', color='skyblue')  # bins 决定分箱数
plt.title('Frequency Distribution of Biomass', fontsize=16)
plt.xlabel('Ht', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#%%
sym90 = pd.read_csv('./DATA/sym90.csv', sep=',') # trmc代码转名称, encoding = 'gb2312'
data1 = pd.merge(df2, sym90, on='TRMC', how='left')
#%%
data2 = data1[['Ht',
        'BeijingNDVI_MAX', 'BJAspect',
       'BJElevation', 'BJSlope', 'BJVHMedian', 'BJVVMedian',
       'TRZD', 'wc2_1_30s_bio_1', 'wc2_1_30s_bio_2', 'wc2_1_30s_bio_3',
       'wc2_1_30s_bio_4', 'wc2_1_30s_bio_5', 'wc2_1_30s_bio_6',
       'wc2_1_30s_bio_7', 'wc2_1_30s_bio_8', 'wc2_1_30s_bio_9',
       'wc2_1_30s_bio_10', 'wc2_1_30s_bio_11', 'wc2_1_30s_bio_12',
       'wc2_1_30s_bio_13', 'wc2_1_30s_bio_14', 'wc2_1_30s_bio_15',
       'wc2_1_30s_bio_16', 'wc2_1_30s_bio_17', 'wc2_1_30s_bio_18',
       'wc2_1_30s_bio_19','SU_SYM90']]

#%% 计算每列的众数
# mode_values = data2.mode().iloc[0]
# # 使用每列的众数填充空值
# data2 = data2.fillna(mode_values)
data2 = data2.dropna(axis=0,how='any')

#%%处理离散数据-哑变量处理
# 将列转换为字符类型
data2['TRZD'] = data2['TRZD'].round().astype('Int64')
data2['TRZD'] = data2['TRZD'].astype(str)

# sub_df['pnf'] = round(sub_df['pnf'])
data3 = pd.get_dummies(
    data2,
    columns=['TRZD','SU_SYM90'],
    prefix=['TRZD','TRMC'],
    prefix_sep="_",
    dummy_na=False,
    drop_first=False)

#%%#####################################################################################################################
###########################################初始化模型筛选汇总结果（未调参数）#################################################
########################################################################################################################
# check version
from pycaret.utils import version
version()
################### Setup ➡️ Compare Models ➡️ Analyze Model ➡️ Prediction ➡️ Save Model ##############################

# This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function. It takes two required parameters: data and target. All the other parameters are optional.
from pycaret.regression import *
#
s = setup(data3, target = 'Ht', session_id = 2024,use_gpu = True,train_size = 0.7)
# check available models  1
all_models = models()

#% Compare Models
best = compare_models()
print(best)

allmodels = pull()
#%%
compare_tree_models = compare_models(include = ['et', 'rf', 'dt', 'gbr', 'xgboost', 'lightgbm', 'catboost','ada','ridge','br'])  #H1
# compare_tree_models = compare_models(include = ['et', 'rf', 'dt', 'gbr', 'xgboost', 'lightgbm', 'catboost','ada','ridge','br'])    #H2
# compare_tree_models = compare_models(include = ['et', 'rf', 'dt', 'gbr', 'xgboost', 'lightgbm', 'catboost','ada','ridge'])         #H3

#%
compare_tree_models_results = pull()
compare_tree_models_results
#%%#####################################################################################################################
###########################################独立检验数据集汇总结果（未调参数）#################################################
########################################################################################################################
trainX = data3[['BeijingNDVI_MAX', 'BJAspect', 'BJElevation', 'BJSlope',
       'BJVHMedian', 'BJVVMedian', 'wc2_1_30s_bio_1', 'wc2_1_30s_bio_2',
       'wc2_1_30s_bio_3', 'wc2_1_30s_bio_4', 'wc2_1_30s_bio_5',
       'wc2_1_30s_bio_6', 'wc2_1_30s_bio_7', 'wc2_1_30s_bio_8',
       'wc2_1_30s_bio_9', 'wc2_1_30s_bio_10', 'wc2_1_30s_bio_11',
       'wc2_1_30s_bio_12', 'wc2_1_30s_bio_13', 'wc2_1_30s_bio_14',
       'wc2_1_30s_bio_15', 'wc2_1_30s_bio_16', 'wc2_1_30s_bio_17',
       'wc2_1_30s_bio_18', 'wc2_1_30s_bio_19', 'TRZD_1', 'TRZD_2', 'TRZD_3',
       'TRMC_ARl', 'TRMC_CLh', 'TRMC_CLl', 'TRMC_FLe', 'TRMC_FLm', 'TRMC_FLt',
       'TRMC_FLu', 'TRMC_GLu', 'TRMC_GRh', 'TRMC_GYp', 'TRMC_LPq', 'TRMC_LVa']]

trainY = data3[['Ht']]
#%%
from sklearn.model_selection import train_test_split, KFold
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, train_size=0.7, random_state=2024)  # 数据集划分

#%%
import pandas as pd
AGEdata = pd.DataFrame(None,columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
#%%# 1 DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=2024)
dt_model.fit( X_train, y_train)
dt_predict = dt_model.predict(X_test)
dt_predict1 = dt_model.predict(X_train)
#% #测试集验证结果
AGE1 = pd.DataFrame([['DecisionTreeRegressor',r2_score(y_train, dt_predict1), sqrt(mean_squared_error(y_train, dt_predict1)),r2_score(y_test, dt_predict),sqrt(mean_squared_error(y_test, dt_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= pd.concat([AGEdata, AGE1], ignore_index=True)
# AGEdata= AGEdata.append(AGE1)
print(AGEdata)

#%%# 2ExtraTreeRegressor
et_model = ExtraTreeRegressor(random_state=2024)
et_model.fit( X_train, y_train)
et_predict = et_model.predict(X_test)
et_predict1 = et_model.predict(X_train)
AGE2 = pd.DataFrame([['ExtraTreeRegressor',r2_score(y_train, et_predict1), sqrt(mean_squared_error(y_train, et_predict1)),r2_score(y_test, et_predict),sqrt(mean_squared_error(y_test, et_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= pd.concat([AGEdata, AGE2], ignore_index=True)
print(AGEdata)

#%% 3XGBRegressor####################model1
xgbr_model = XGBRegressor(random_state=2024)
xgbr_model.fit(X_train, y_train)
xgbr_predict = xgbr_model.predict(X_test)
xgbr_predict1 = xgbr_model.predict(X_train)

AGE4 = pd.DataFrame([['XGBRegressor',r2_score(y_train, xgbr_predict1), sqrt(mean_squared_error(y_train, xgbr_predict1)),r2_score(y_test, xgbr_predict),sqrt(mean_squared_error(y_test, xgbr_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= pd.concat([AGEdata, AGE4], ignore_index=True)
print(AGEdata)
#%% 4 HistGradientBoostingRegressor#############
HistGB_model = HistGradientBoostingRegressor(random_state=2024)
HistGB_model.fit(X_train, y_train)
HistGB_predict = HistGB_model.predict(X_test)
HistGB_predict1 = HistGB_model.predict(X_train)
AGE5 = pd.DataFrame([['HistGradientBoostingRegressor',r2_score(y_train, HistGB_predict1), sqrt(mean_squared_error(y_train, HistGB_predict1)),r2_score(y_test, HistGB_predict),sqrt(mean_squared_error(y_test, HistGB_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= pd.concat([AGEdata, AGE5], ignore_index=True)
print(AGEdata)
#%%# 5 LGBMRegressor###################model3
ligthgbmc_model = LGBMRegressor(random_state=2024)
ligthgbmc_model.fit(X_train, y_train)
ligthgbmc_predict = ligthgbmc_model.predict(X_test)
ligthgbmc_predict1 = ligthgbmc_model.predict(X_train)
AGE6 = pd.DataFrame([['LGBMRegressor',r2_score(y_train, ligthgbmc_predict1), sqrt(mean_squared_error(y_train, ligthgbmc_predict1)),r2_score(y_test, ligthgbmc_predict),sqrt(mean_squared_error(y_test, ligthgbmc_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= pd.concat([AGEdata, AGE6], ignore_index=True)
print(AGEdata)

#%%6 RandomForestRegressor####################model5
rf_model = RandomForestRegressor(random_state=2024)
rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict(X_test)
rf_predict1 = rf_model.predict(X_train)
AGE7 = pd.DataFrame([['RandomForestRegressor',r2_score(y_train, rf_predict1), sqrt(mean_squared_error(y_train, rf_predict1)),r2_score(y_test, rf_predict),sqrt(mean_squared_error(y_test, rf_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= pd.concat([AGEdata, AGE7], ignore_index=True)
print(AGEdata)

#%%将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename = r"./model/RF_HT.model",value=rf_model)
#%%7 GradientBoostingRegressor################model4
gb_model = GradientBoostingRegressor(random_state=2024)
gb_model.fit(X_train, y_train)
gb_predict = gb_model.predict(X_test)
gb_predict1 = gb_model.predict(X_train)

AGE9 = pd.DataFrame([['GradientBoostingRegressor',r2_score(y_train, gb_predict1), sqrt(mean_squared_error(y_train, gb_predict1)),r2_score(y_test, gb_predict),sqrt(mean_squared_error(y_test, gb_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= pd.concat([AGEdata, AGE9], ignore_index=True)
print(AGEdata)
#%%8 AdaBoostRegressor###############
AdaB_model = AdaBoostRegressor(random_state=2024)
AdaB_model.fit(X_train, y_train)
AdaB_predict = AdaB_model.predict(X_test)
AdaB_predict1 = AdaB_model.predict(X_train)

AGE11 = pd.DataFrame([['AdaBoostRegressor',r2_score(y_train, AdaB_predict1), sqrt(mean_squared_error(y_train, AdaB_predict1)),r2_score(y_test, AdaB_predict),sqrt(mean_squared_error(y_test, AdaB_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= pd.concat([AGEdata, AGE11], ignore_index=True)
print(AGEdata)

#%%12 KNeighborsRegressor###############
KN_model = KNeighborsRegressor(n_jobs=-1)
KN_model.fit(X_train, y_train)
KN_predict = KN_model.predict(X_test)
KN_predict1 = KN_model.predict(X_train)
AGE12 = pd.DataFrame([['KNeighborsRegressor',r2_score(y_train, KN_predict1), sqrt(mean_squared_error(y_train, KN_predict1)),r2_score(y_test, KN_predict),sqrt(mean_squared_error(y_test, KN_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
# AGEdata = AGEdata.append(AGE12)
AGEdata= pd.concat([AGEdata, AGE12], ignore_index=True)
print(AGEdata)

#%%# 9CatBoostRegressor#############model2
Cat_model = CatBoostRegressor(random_state=2024, verbose=False)
Cat_model.fit(X_train, y_train)
Cat_predict = Cat_model.predict(X_test)
Cat_predict1 = Cat_model.predict(X_train)
AGE13 = pd.DataFrame([['CatBoostRegressor',r2_score(y_train, Cat_predict1), sqrt(mean_squared_error(y_train, Cat_predict1)),r2_score(y_test, Cat_predict),sqrt(mean_squared_error(y_test, Cat_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= pd.concat([AGEdata, AGE13], ignore_index=True)
print(AGEdata)

#%%10 deepforest-CascadeForestRegressor
# from deepforest import CascadeForestRegressor
#
# deeprf_model = CascadeForestRegressor(random_state=2024)
# deeprf_model.fit(X_train, y_train)
# deeprf_predict = deeprf_model.predict(X_test)
# deeprf_predict1 = deeprf_model.predict(X_train)
# AGE10 = pd.DataFrame([['CascadeForestRegressor',r2_score(y_train, deeprf_predict1), sqrt(mean_squared_error(y_train, deeprf_predict1)),r2_score(y_test, deeprf_predict),sqrt(mean_squared_error(y_test, deeprf_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
# AGEdata= AGEdata.append(AGE10)
# print(AGEdata)
#%%#####################################################################################################################
###########################################独立检验数据集进行调参过程#################################################
########################################################################################################################
# GBDT + Optuna#########################################################################################################
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),#大容易过拟合
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001,0.005,0.01,0.05,0.1]),#大容易过拟合
        'n_estimators': trial.suggest_int('n_estimators', 4000, 5000),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),#小容易过拟合
        # 'max_features':trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),#大容易过拟合
        'random_state': 2024
    }

    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params

#%% LGBM + Optuna#########################################################################################################
def objective(trial):
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 11, 333),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.005, 0.1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 7000, 8000),
        'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
        'cat_l2': trial.suggest_int('cat_l2', 1, 20),
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
        'cat_feature': trial.suggest_int('cat_feature', 10, 60),
        'n_jobs': -1,
        'force_col_wise': 'true',
        'random_state': 2024,
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params


#%% RF + Optuna############################################MODEL4
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3,18),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_features':trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 2024
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params

#%% #######################CatBoost + Optuna######################MODEL2
import time
start = time.process_time()
def objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01, 0.05]),
        'iterations': trial.suggest_int('iterations', 5000, 8000),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'random_seed': 2024
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=2022, verbose=False)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
# study.best_params
end = time.process_time()
print('CPU Times ', end-start)
study.best_params

#%% #######################CascadeForestRegressor + Optuna######################MODEL2
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_layers': trial.suggest_int('max_layers', 3, 10),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 2024
    }
    model = CascadeForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params

#%%####################### HistGradientBoostingRegressor + Optuna ############################################
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),#大容易过拟合
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001,0.005,0.01,0.05,0.1]),#大容易过拟合
        'max_leaf_nodes':trial.suggest_int('max_leaf_nodes', 30, 40),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 15, 25),#大容易过拟合
        'random_state': 2024
    }

    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params

#%%############################################Xgboost + Optuna#####################################################
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01]),
        'n_estimators': trial.suggest_int('n_estimators', 2000, 8000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_float('gamma', 0.0001, 1.0, log=True),
        'reg_alpha': trial.suggest_float('alpha', 0.0001, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lambda', 0.0001, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'random_state': 2024
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=2024, verbose=False)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params


#%%############################################AdaBoost + Optuna#####################################################
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'loss': trial.suggest_categorical('loss', ['linear', 'exponential']),
        'base_estimator': DecisionTreeRegressor(max_depth=trial.suggest_int('max_depth', 1, 5)),
        'random_state': 2024
    }
    model = AdaBoostRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params
########################################################################################################################
####################################################对已经调好的参数模型，统计其变量值########################################
########################################################################################################################
#%%统计调参后结果
import pandas as pd
INDEX_TRAIN = pd.DataFrame(None,columns=['Model','R2','RMSE','MSE','MAE','ME'])
INDEX_TEST = pd.DataFrame(None,columns=['Model','R2','RMSE','MSE','MAE','ME'])

#%%1 CatBoostRegressor
paras = {'depth': 10,
 'learning_rate': 0.05,
 'iterations': 5532,
 'max_bin': 229,
 'min_data_in_leaf': 9,
 'l2_leaf_reg': 0.1820485748656637,
 'subsample': 0.7058903419805836,
         'random_state': 2024}


model1 = CatBoostRegressor(**paras)
model1.fit(X_train, y_train,eval_set=[(X_test, y_test)],plot=True)

pred_train = model1.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)
#%
pred_test = model1.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN1 = pd.DataFrame([['CatBoostRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST1 = pd.DataFrame([['CatBoostRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#%%
INDEX_TRAIN= pd.concat([INDEX_TRAIN, INDEX_TRAIN1], ignore_index=True)
INDEX_TEST= pd.concat([INDEX_TEST, INDEX_TEST1], ignore_index=True)
print(INDEX_TRAIN)
print(INDEX_TEST)

#%%
#将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename = r"./model/HT_catboost1225.model",value=model1)

#%%2 HistGradientBoostingRegressor
paras = {'max_depth': 6,
         'learning_rate': 0.1,
         'max_leaf_nodes': 36,
         'min_samples_leaf': 15,
        'random_state': 2024}

model2 = HistGradientBoostingRegressor(**paras)
model2.fit(X_train, y_train)


pred_train = model2.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train,)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
ME_train = np.mean(y_train- pred_train)

pred_test = model2.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN2 = pd.DataFrame([['HistGradientBoostingRegressor',R2_train,RMSE_train,MSE_train,MAE_train,ME_train]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
INDEX_TEST2 = pd.DataFrame([['HistGradientBoostingRegressor',R2_test,RMSE_test,MSE_test,MAE_test,ME_test]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
#
INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN2)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST2)
print(INDEX_TRAIN)
print(INDEX_TEST)


#%%3 GradientBoostingRegressor
paras = {'max_depth': 8,
         'learning_rate': 0.005,
         'n_estimators': 4750,
         'subsample': 0.7832677020684471,
         'min_samples_split': 4,
         'random_state': 2024}

model3 = GradientBoostingRegressor(**paras)
model3.fit(X_train, y_train)
#%%
pred_train = model3.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
ME_train = 0

pred_test = model3.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
ME_test = 0

INDEX_TRAIN3 = pd.DataFrame([['GradientBoostingRegressor',R2_train,RMSE_train,MSE_train,MAE_train,ME_train]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
INDEX_TEST3 = pd.DataFrame([['GradientBoostingRegressor',R2_test,RMSE_test,MSE_test,MAE_test,ME_test]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
#
INDEX_TRAIN= pd.concat([INDEX_TRAIN, INDEX_TRAIN3], ignore_index=True)
INDEX_TEST= pd.concat([INDEX_TEST, INDEX_TEST3], ignore_index=True)
print(INDEX_TRAIN)
print(INDEX_TEST)

#%%4 RandomForestRegressor
paras = {'max_depth': 6,
         'n_estimators': 100,
         'max_features': 1.0,
         'min_samples_split': 2,
         'min_samples_leaf': 1,
         'random_state': 2024}

model4 = RandomForestRegressor(**paras)
model4.fit(X_train, y_train)


pred_train = model4.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)

pred_test = model4.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)
#%%
INDEX_TRAIN4 = pd.DataFrame([['RandomForestRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST4 = pd.DataFrame([['RandomForestRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#
INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN4)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST4)
print(INDEX_TRAIN)
print(INDEX_TEST)

#%%5 LGBMRegressor
paras = {'reg_alpha': 1.8142103433886785,
 'reg_lambda': 3.9617790715268013,
 'num_leaves': 262,
 'min_child_samples': 10,
 'max_depth': 15,
 'learning_rate': 0.05,
 'colsample_bytree': 0.49887349659988267,
 'n_estimators': 7493,
 'cat_smooth': 95,
 'cat_l2': 5,
 'min_data_per_group': 64,
 'cat_feature': 57,
 'random_state': 2024}

# model5 = joblib.load(filename="./model/newh1.model")#加载模型
model5 = LGBMRegressor(**paras)
model5.fit(X_train, y_train)
#%%
model5.feature_importances_
#%%保存重要性
columns = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7',
       'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14',
       'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age', 'VH', 'VV',
       'tchd', 'aspect', 'elevation', 'slope', 'NDVI_MAX', 'TRZD_1', 'TRZD_2',
       'TRZD_3', 'TRMC_ACf', 'TRMC_ACh', 'TRMC_ACp', 'TRMC_ACu', 'TRMC_ALf',
       'TRMC_ALh', 'TRMC_ANh', 'TRMC_ANu', 'TRMC_ARb', 'TRMC_ARc', 'TRMC_ARh',
       'TRMC_ATc', 'TRMC_CHh', 'TRMC_CHl', 'TRMC_CMc', 'TRMC_CMd', 'TRMC_CMe',
       'TRMC_CMg', 'TRMC_CMo', 'TRMC_CMx', 'TRMC_FLc', 'TRMC_FLe', 'TRMC_FRh',
       'TRMC_FRx', 'TRMC_GLm', 'TRMC_GLt', 'TRMC_GRh', 'TRMC_GYp', 'TRMC_KSh',
       'TRMC_KSk', 'TRMC_LPe', 'TRMC_LPm', 'TRMC_LVa', 'TRMC_LVg', 'TRMC_LVh',
       'TRMC_LVj', 'TRMC_LVk', 'TRMC_LVx', 'TRMC_LXa', 'TRMC_LXf', 'TRMC_NTu',
       'TRMC_PHh', 'TRMC_PLe', 'TRMC_RGc', 'TRMC_RGd', 'TRMC_RGe', 'TRMC_SCk',
       'TRMC_SNk', 'TRMC_VRd', 'TRMC_WR', 'PNF_1', 'PNF_2']
df = pd.DataFrame()
df['feature name'] = columns
df['importance'] = list(model5.feature_importances_)
#%%
pred_train = model5.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)

pred_test = model5.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)
#%%
INDEX_TRAIN5 = pd.DataFrame([['LGBMRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST5 = pd.DataFrame([['LGBMRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#%%
INDEX_TRAIN= pd.concat([INDEX_TRAIN, INDEX_TRAIN5], ignore_index=True)
INDEX_TEST= pd.concat([INDEX_TEST, INDEX_TEST5], ignore_index=True)

print(INDEX_TRAIN)
print(INDEX_TEST)
#%%
preht_test = y_test
preht_test['preHT'] =  pred_test

#%%
#将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename = r"./model/HT_lightgbm.model",value=model5)

#%%6 XGBRegressor
paras = {'max_depth': 10,
 'learning_rate': 0.01,
 'n_estimators': 7203,
 'min_child_weight': 1,
 'gamma': 0.06728229770511852,
 'alpha': 0.21304909691832852,
 'lambda': 0.00013819888953716042,
 'colsample_bytree': 0.4374333788046241,
 'subsample': 0.6068106161836021,
 'random_state': 2024}


model6 = XGBRegressor(**paras)
model6.fit(X_train, y_train)

pred_train = model6.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)
#%
pred_test = model6.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN6 = pd.DataFrame([['XGBRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST6 = pd.DataFrame([['XGBRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#
INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN6)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST6)
print(INDEX_TRAIN)
print(INDEX_TEST)



#%%zone7 AdaBoostRegressor
paras = {'n_estimators': 239,
         'learning_rate': 0.06425924663287144,
         'loss': 'linear',
         'base_estimator': DecisionTreeRegressor(max_depth=5),
         'random_state': 2024}

model7 = AdaBoostRegressor(**paras)
model7.fit(X_train, y_train)


pred_train = model7.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
ME_train = np.mean(y_train- pred_train)

pred_test = model7.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN7 = pd.DataFrame([['AdaBoostRegressor',R2_train,RMSE_train,MSE_train,MAE_train,ME_train]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
INDEX_TEST7 = pd.DataFrame([['AdaBoostRegressor',R2_test,RMSE_test,MSE_test,MAE_test,ME_test]],columns=['Model','R2','RMSE','MSE','MAE','ME'])

INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN7)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST7)
print(INDEX_TRAIN)
print(INDEX_TEST)

#%%zone8 CascadeForestRegressor
paras = {'n_estimators': 239,
         'learning_rate': 0.06425924663287144,
         'loss': 'linear',
         'base_estimator': DecisionTreeRegressor(max_depth=5),
         'random_state': 2024}
model8 = CascadeForestRegressor(**paras)
model8.fit(X_train, y_train)


pred_train = model8.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
ME_train = np.mean(y_train- pred_train)

pred_test = model8.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN8 = pd.DataFrame([['CascadeForestRegressor',R2_train,RMSE_train,MSE_train,MAE_train,ME_train]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
INDEX_TEST8 = pd.DataFrame([['CascadeForestRegressor',R2_test,RMSE_test,MSE_test,MAE_test,ME_test]],columns=['Model','R2','RMSE','MSE','MAE','ME'])

INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN8)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST8)
print(INDEX_TRAIN)
print(INDEX_TEST)

#%%##############################################基于最优结果计算其固定效应预测值##############################################
#获取原始数据的2个字段
data2A = data22[['HT','REGION']]
trainX1 = data2A[['REGION']].values
trainY1 = data2A[['HT']].values

from sklearn.model_selection import train_test_split, KFold
X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX1, trainY1, train_size=2/3, random_state=2024)  # 数据集划分
#%% X_train
loaded_model = joblib.load(filename="model/lightgbm_ht_0620.model")#加载模型
preHT = loaded_model.predict(X_test)# 使用模型对测试数据进行预测

#%% 将ndarray转换为DataFrame
data2A = pd.DataFrame({'REGION': X_test1.flatten(), 'HT': y_test1.flatten(), 'fixedHT': preHT})

data2A.to_csv("./DATA/R_HT_new.csv", index=False,encoding='utf-8-sig')

########################################################################################################################
####################################################基于lem4py包，实现混合效应模型构建########################################
#%%#####################################################################################################################
'''
此部分实现用R语言进行运行，得出固定效应的参数值以及随机效应参数（按植被区对应），进行下一下预测。各参数值如下所示：
固定效应截距项：fixed_Intercept         (1个值）
固定效应preH1对应斜率参数：fixed_preH1   (1个值）
随机效应截距项：random_Intercept        (8个值，每个植被区对应一个值）
'''

########################################################################################################################
###################################################测试集进行检验########################################
#%%#####################################################################################################################
#ht mlme
loaded_model = joblib.load(filename="./model/lightgbm_ht_0620.model")#加载模型, encoding = 'gb2312'
preHT = loaded_model.predict(X_test)# 使用模型对测试数据进行预测

#%%取test数据集X_test2
DataHT = pd.read_csv('./DATA/R_HT_new.csv', sep=',')
ParasHT = pd.read_csv('./DATA/HT_new.csv', sep=',', encoding = 'gb2312')
HTdat = pd.merge(DataHT, ParasHT, on='REGION', how='left')
#%%
# trainX2 = H1dat[['REGION','InterceptH1','fixedH1']].values
# trainY2 = HTdat[['H1']].values
# X_train2, X_test2, y_train2, y_test2 = train_test_split(trainX2, trainY2, train_size=2/3, random_state=2024)  # 数据集划分
#%%
# data2B = pd.DataFrame({'REGION': X_test2[:,0],'H3': y_test2.flatten(), 'InterceptH3': X_test2[:,1], 'fixedH3': X_test2[:,2],'preH3': preH3})
HTdat['predict_HT'] = HTdat['fixedHT']*HTdat['fixedHT_p'] + HTdat['InterceptHT']
HTdat['R'] = HTdat['HT']- HTdat['predict_HT']
HTdat = HTdat[(HTdat['R'] >= -35) & (HTdat['R'] <= 35)]
#%%
# pred_test = data2B[['predict_H3']].values
R2_test = r2_score(HTdat['HT'], HTdat['predict_HT'])
RMSE_test = mean_squared_error(HTdat['HT'], HTdat['predict_HT'], squared=False)
MSE_test = mean_squared_error(HTdat['HT'], HTdat['predict_HT'])
rRMSE_test = RMSE_test/HTdat['HT'].mean()
MAE_test = mean_absolute_error(HTdat['HT'], HTdat['predict_HT'])
#%%
print(R2_test,MSE_test,RMSE_test,rRMSE_test,MAE_test)

#%%h3 mlme
loaded_model = joblib.load(filename="./model/lightgbm_h3_0509.model")#加载模型, encoding = 'gb2312'
preH3 = loaded_model.predict(X_test)# 使用模型对测试数据进行预测

#%%取test数据集X_test2
DataH3 = pd.read_csv('./DATA/R_H3_new.csv', sep=',')
ParasH3 = pd.read_csv('./DATA/H3_new.csv', sep=',', encoding = 'gb2312')
H3dat = pd.merge(DataH3, ParasH3, on='REGION', how='left')
#%%
# trainX2 = H1dat[['REGION','InterceptH1','fixedH1']].values
# trainY2 = HTdat[['H1']].values
# X_train2, X_test2, y_train2, y_test2 = train_test_split(trainX2, trainY2, train_size=2/3, random_state=2024)  # 数据集划分
#%%
# data2B = pd.DataFrame({'REGION': X_test2[:,0],'H3': y_test2.flatten(), 'InterceptH3': X_test2[:,1], 'fixedH3': X_test2[:,2],'preH3': preH3})
H3dat['predict_H3'] = H3dat['fixedH3']*H3dat['fixedH3_p'] + H3dat['InterceptH3']
#%%
# pred_test = data2B[['predict_H3']].values
R2_test = r2_score(H3dat['H3'], H3dat['predict_H3'])
RMSE_test = mean_squared_error(H3dat['H3'], H3dat['predict_H3'], squared=False)
MSE_test =  mean_squared_error(H3dat['H3'], H3dat['predict_H3'])
rRMSE_test = RMSE_test/H3dat['H3'].mean()
MAE_test = mean_absolute_error(H3dat['H3'], H3dat['predict_H3'])
#%%
print(R2_test,MSE_test,RMSE_test,rRMSE_test,MAE_test)

#%%h1 mlme***************************************************************************************
loaded_model = joblib.load(filename="./model/lightgbm_h1_0509.model")#加载模型, encoding = 'gb2312'
preH1 = loaded_model.predict(X_test)# 使用模型对测试数据进行预测

#%%取test数据集X_test2
DataH1 = pd.read_csv('./DATA/R_H1_new.csv', sep=',')
ParasH1 = pd.read_csv('./DATA/H1_new.csv', sep=',', encoding = 'gb2312')
H1dat = pd.merge(DataH1, ParasH1, on='REGION', how='left')
#%%
# trainX2 = H1dat[['REGION','InterceptH1','fixedH1']].values
# trainY2 = HTdat[['H1']].values
# X_train2, X_test2, y_train2, y_test2 = train_test_split(trainX2, trainY2, train_size=2/3, random_state=2024)  # 数据集划分
#%%
# data2B = pd.DataFrame({'REGION': X_test2[:,0],'H3': y_test2.flatten(), 'InterceptH3': X_test2[:,1], 'fixedH3': X_test2[:,2],'preH3': preH3})
H1dat['predict_H1'] = H1dat['fixedH1']*H1dat['fixedH1_p'] + H1dat['InterceptH1']
#%%

R2_test = r2_score(H1dat['H1'], H1dat['predict_H1'])
RMSE_test = mean_squared_error(H1dat['H1'], H1dat['predict_H1'], squared=False)
MSE_test = mean_squared_error(H1dat['H1'], H1dat['predict_H1'])
rRMSE_test = RMSE_test/H1dat['H1'].mean()
MAE_test = mean_absolute_error(H1dat['H1'], H1dat['predict_H1'])
#%%
print(R2_test,MSE_test,RMSE_test,rRMSE_test,MAE_test)

#%%h1 ml***********************************************
loaded_model = joblib.load(filename="./model/lightgbm_h1_0509.model")#加载模型, encoding = 'gb2312'
pred_test = loaded_model.predict(X_test)# 使用模型对测试数据进行预测

R2_test = r2_score(y_test, pred_test)
RMSE_test = mean_squared_error(y_test, pred_test, squared=False)
MSE_test = mean_squared_error(y_test, pred_test)
rRMSE_test = RMSE_test/y_test.mean()
MAE_test = mean_absolute_error(y_test, pred_test)


print(R2_test,MSE_test,RMSE_test,rRMSE_test,MAE_test)

#%%
H1dat.to_excel("./DATA/H1dat.xlsx")

#%%
# 所有输入数据的tif
all_tiff_directory = r"G:\TH\Predict\zone30\preTIF"#所有tiff位置,预测数据集
# 获取文件夹中的tif文件路径列表
import os
tiff_files1 = [file for file in os.listdir(all_tiff_directory) if file.endswith('.tif')]

#%%
# 将列表转换为数据框的一列
df = pd.DataFrame({'Column_Name': tiff_files1})



################################优势高探索评价###################################

#%%read data
htdf1 = pd.read_csv('./DATA/LUO150_X.csv', sep=',', encoding = 'gb2312')
sym90 = pd.read_csv('./DATA/sym90.csv', sep=',')#trmc代码转名称
htdata1 = pd.merge(htdf1, sym90, on='trmc', how='left')

#%%
htdata2 = htdata1[[
        'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
        'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'AGE',
        'VH', 'VV', 'tchd', 'trzd', 'aspect', 'elevation', 'slope', 'pnf', 'NDVI_MAX', 'SU_SYM90']]

#%%处理离散数据-哑变量处理
# 将列转换为字符类型

htdata2['trzd'] = htdata2['trzd'].round().astype('Int64')
htdata2['trzd'] = htdata2['trzd'].astype(str)
htdata2['pnf'] = htdata2['pnf'].round().astype('Int64')
htdata2['pnf'] = htdata2['pnf'].astype(str)
# sub_df['pnf'] = round(sub_df['pnf'])
htdata3 = pd.get_dummies(
    htdata2,
    columns=['trzd','SU_SYM90','pnf'],
    prefix=['TRZD','TRMC','PNF'],
    prefix_sep="_",
    dtype=int,
    dummy_na=False,
    drop_first=False)

# htdata3['TRMC_GYp'] =0
# htdata3['TRMC_KSk'] =0
# htdata3['TRMC_LVj'] =0
# htdata3['TRMC_SCk'] =0
# htdata3['TRMC_SNk'] =0

columns_to_update = ['TRMC_ACf', 'TRMC_ACh', 'TRMC_ACp', 'TRMC_ACu', 'TRMC_ALf', 'TRMC_ANh', 'TRMC_ARh', 'TRMC_CMg', 'TRMC_CMo', 'TRMC_CMx', 'TRMC_FLe', 'TRMC_FRh', 'TRMC_FRx', 'TRMC_GLt', 'TRMC_GYp', 'TRMC_KSh', 'TRMC_KSk', 'TRMC_LVx', 'TRMC_LXa', 'TRMC_LXf', 'TRMC_NTu', 'TRMC_PLe', 'TRMC_RGc', 'TRMC_RGd', 'TRMC_SCk', 'TRMC_SNk', 'TRMC_VRd', 'TRMC_WR']
# 将指定列的值设置为 0
htdata3[columns_to_update] = 0
#%%
X_ht = htdata3[['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7',
       'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14',
       'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'AGE', 'VH', 'VV',
       'tchd', 'aspect', 'elevation', 'slope', 'NDVI_MAX', 'TRZD_1', 'TRZD_2',
       'TRZD_3', 'TRMC_ACf', 'TRMC_ACh', 'TRMC_ACp', 'TRMC_ACu', 'TRMC_ALf',
       'TRMC_ALh', 'TRMC_ANh', 'TRMC_ANu', 'TRMC_ARb', 'TRMC_ARc', 'TRMC_ARh',
       'TRMC_ATc', 'TRMC_CHh', 'TRMC_CHl', 'TRMC_CMc', 'TRMC_CMd', 'TRMC_CMe',
       'TRMC_CMg', 'TRMC_CMo', 'TRMC_CMx', 'TRMC_FLc', 'TRMC_FLe', 'TRMC_FRh',
       'TRMC_FRx', 'TRMC_GLm', 'TRMC_GLt', 'TRMC_GRh', 'TRMC_GYp', 'TRMC_KSh',
       'TRMC_KSk', 'TRMC_LPe', 'TRMC_LPm', 'TRMC_LVa', 'TRMC_LVg', 'TRMC_LVh',
       'TRMC_LVj', 'TRMC_LVk', 'TRMC_LVx', 'TRMC_LXa', 'TRMC_LXf', 'TRMC_NTu',
       'TRMC_PHh', 'TRMC_PLe', 'TRMC_RGc', 'TRMC_RGd', 'TRMC_RGe', 'TRMC_SCk',
       'TRMC_SNk', 'TRMC_VRd', 'TRMC_WR', 'PNF_1', 'PNF_2']].values

#%% X_train
loaded_model = joblib.load(filename="model/lightgbm_hT_0620.model")#加载模型
preHT = loaded_model.predict(X_ht)# 使用模型对测试数据进行预测
htdf1["preHT"] = preHT

# 1. 决策树
# 原理:决策树是一种递归地将数据划分成小的子集的算法，直到每个子集中的数据点都属于同一个类。
# 它使用特征的阈值来进行分裂，选择使得每次分裂后的子集尽可能纯的特征和阈值。
#%%
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import tree
import graphviz
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#%%read data
# 读取数据
madf0 = pd.read_csv('./DATA/MA220C.csv', sep=',', encoding='gb2312')
# 删除缺失值
madf1 = madf0.dropna()
# 选择需要的列
madf2 = madf1[['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8',
               'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16',
               'bio_17', 'bio_18', 'bio_19', 'trmc', 'trzd', 'tchd',
               'aspect', 'elevation', 'slope', 'class']]

#%% 将 'trzd' 列和 'tchd' 列转换为整数然后转换为字符串
madf2.loc[:, 'trzd']  = madf2['trzd'].round().astype('Int64').astype(str)
madf2.loc[:, 'tchd']  = madf2['tchd'].round().astype('Int64').astype(str)
# 定义特征和目标变量
X = madf2.drop('class', axis=1)
y = madf2['class']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

#%%参数寻优
# 定义目标函数
def objective(trial):
    # 定义需要优化的参数
    max_depth = trial.suggest_int('max_depth', 1, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # 创建决策树模型
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=2024
    )
    # 训练模型
    clf.fit(X_train, y_train)
    # 预测
    y_pred = clf.predict(X_test)
    # 计算准确率
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1
# 创建Optuna的研究对象
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
# 输出最佳参数
print("Best parameters: ", study.best_params)
print("Best F1: ", study.best_value)

#%%使用决策树算法进行分类
paras = {'max_depth': 9, 'min_samples_split': 8, 'min_samples_leaf': 5,'random_state':2024}
clf_best = DecisionTreeClassifier(**paras)
clf_best.fit(X_train, y_train)

#%% 打印生成的规则
tree_rules = export_text(clf_best, feature_names=list(X.columns))
print(tree_rules)
#%% 可视化决策树
dot_data = tree.export_graphviz(clf_best, out_file=None,
                                feature_names=X.columns,
                                class_names=y.unique(),
                                filled=True, rounded=True,
                                special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree", format='png')  # 保存为图像文件
graph.view()  # 打开默认图像查看器显示决策树
#%% 对测试数据进行预测
y_pred = clf_best.predict(X_test)
# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# 计算精确度
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision}")
# 计算召回率
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall}")
# 计算F1分数
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1}")
# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("计算混淆矩阵Confusion Matrix:")
print(conf_matrix)
# 输出分类报告
print("分类报告:")
print(classification_report(y_test, y_pred))

