# -*- coding: utf-8 -*-
# @Time    : 2024/3/21 11:30
# @Author  : ChenYuling
# @FileName: CHN_h.py
# @Software: PyCharm
# @Describe：数据处理部分：计算各个CHM的树高数据

#%%
import os
import pandas as pd

csv_directory = r"F:\DB\TH\TrainDATA\CHMVersion1.2\data\Z49\csv"  #----------------------------------------------------------------------------->gaidongchu
# 存储样地属性的列表
plot_THs = []
#%树高相关系列值计算计算
sub_tiff_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]
for subcsv_file in sub_tiff_files:
    subcsv_path = os.path.join(csv_directory, subcsv_file)
    subcsv_name = os.path.basename(subcsv_path).split('.')[0][:-len("_CHM分割")]
    print(subcsv_name)
    data = pd.read_csv(subcsv_path, sep=',',encoding = 'gb2312')#
    if(len(data)==0):
        continue
    data.columns = ['TreeID', 'X', 'Y', 'TreeHeight', 'CW', 'CW_SN', 'CW_EW', 'S']
    if(len(data)==0):
        continue
        #林分平均高计算
    # df = pysqldf(
    #     """ SELECT  AVG(TreeHeight) AS H1,SUM(TreeHeight * TreeHeight)/SUM(TreeHeight) AS H2,SUM(TreeHeight * TreeHeight * TreeHeight)/SUM(TreeHeight * TreeHeight) AS H3,COUNT(TreeID) AS N ,AVG(CW) AS CW,AVG(CW_EW) AS CW_EW,AVG(CW_SN) AS CW_SN,SUM(S) AS S from data """)
    H1 = data['TreeHeight'].mean()
    H2 = (data['TreeHeight']*data['TreeHeight']).sum()/data['TreeHeight'].sum()
    H3 = (data['TreeHeight']*data['TreeHeight']*data['TreeHeight']).sum()/(data['TreeHeight']*data['TreeHeight']).sum()
    H4 = (data['TreeHeight'] * data['CW']).sum() / data['CW'].sum()
    N = len(data)
    CW_mean = data['CW'].mean()
    CWSN_mean = data['CW_SN'].mean()
    CWEW_mean = data['CW_EW'].mean()
    CWArea_sum = data['S'].sum()
    #筛选出每组前3条最大树高数据集
    dataTOP3 = (data.sort_values(by='TreeHeight', ascending=False)).head(3)
    #分组计算前3最高树的平均值为优势高数据
    HT = dataTOP3['TreeHeight'].mean()
    #将平均高几个指标进行合并
    plot_THs.append((subcsv_name, H1, H2, H3, H4, N, CW_mean, CWSN_mean, CWEW_mean, CWArea_sum, HT))

#%% 创建数据框
dfH = pd.DataFrame(plot_THs, columns=['name', 'H1', 'H2', 'H3', 'H4', 'N', 'CW_mean', 'CWSN_mean', 'CWEW_mean', 'CWArea_sum', 'HT'])
# % 保存数据框为 CSV 文件
output_file = "F:\DB\TH\TrainDATA\CHMVersion1.2\data\Z49h_3.csv"  # 输出文件路径-------------------------------------------------------------------->gaidongchu
dfH.to_csv(output_file, index=False)

#%%合并数据
import pandas as pd
csv_directory1 = r"F:\DB\TH\TrainDATA\CHMVersion1.2\data\Z49h.csv" ######---------------------------------------------------------------------->gaidongchu
csv_directory2 = r"F:\DB\TH\TrainDATA\CHMVersion1.2\data\Z49.csv" ######------------------------------------------------------------------->gaidongchu
xy = pd.read_csv(csv_directory2, sep=',')#,encoding = 'gb2312'
H = pd.read_csv(csv_directory1, sep=',')#,encoding = 'gb2312'
merged_xyH = pd.merge(xy, H, on='name', how='inner')

#%%
output_file = "F:\DB\TH\TrainDATA\CHMVersion1.2\data\Zone49.csv"  # 输出文件路径-------------------------------------------------------------------->gaidongchu
merged_xyH.to_csv(output_file, index=False)

