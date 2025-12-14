# -*- coding: utf-8 -*-
# @Time    : 2023/10/24 19:52
# @Author  : ChenYuling
# @FileName: PlotDATA_pysqldf.py
# @Software: PyCharm
# @Describe：数据处理部分：计算林分高数据
#%%
import os
from pandasql import sqldf
pysqldf = lambda q:sqldf(q,globals())
import pandas as pd
csv_directory = r"F:\DB\CHM\zhejiangnonglin\csv"
plot_directory = r"F:\DB\CHM\zhejiangnonglin\plot"
# 创建输出目录（如果不存在）
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
HTplot_directory = r"F:\DB\CHM\zhejiangnonglin\HTplot"
# 创建输出目录（如果不存在）
if not os.path.exists(HTplot_directory):
    os.makedirs(HTplot_directory)


#%% 获取文件夹中的tif文件路径列表 算术平均高和加权平均高计算
sub_tiff_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]
for subcsv_file in sub_tiff_files:
    subcsv_path = os.path.join(csv_directory, subcsv_file)
    subcsv_name = os.path.basename(subcsv_path).split('.')[0]
    data = pd.read_csv(subcsv_path, sep=',')
    df = pysqldf(""" SELECT  Id,PLOTX,PLOTY,AVG(TreeHeight) AS H1,SUM(TreeHeight * TreeHeight)/SUM(TreeHeight) AS H2,SUM(TreeHeight * TreeHeight * TreeHeight)/SUM(TreeHeight * TreeHeight) AS H3,COUNT(TreeID) AS N ,SUM(Area) as S from data GROUP BY  Id,PLOTX,PLOTY""")

    # 创建保存的csv名及位置
    subplot_name = str(subcsv_name) + ".csv"
    subplot_path = os.path.join(plot_directory, subplot_name)
    df.to_csv(subplot_path)

#%% +优势高计算
sub_tiff_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]
for subcsv_file in sub_tiff_files:
    subcsv_path = os.path.join(csv_directory, subcsv_file)
    subcsv_name = os.path.basename(subcsv_path).split('.')[0]
    data = pd.read_csv(subcsv_path, sep=',')
    #林分平均高计算
    df = pysqldf(""" SELECT  Id,PLOTX,PLOTY,AVG(TreeHeight) AS H1,SUM(TreeHeight * TreeHeight)/SUM(TreeHeight) AS H2,SUM(TreeHeight * TreeHeight * TreeHeight)/SUM(TreeHeight * TreeHeight) AS H3,COUNT(TreeID) AS N ,SUM(Area) as S from data GROUP BY  Id,PLOTX,PLOTY""")
    #筛选出每组前3条最大树高数据集
    df_HT_LIST = pysqldf(""" SELECT  d1.Id,d1.PLOTX,d1.PLOTY,d1.TreeHeight from data  d1 left join data  d2 on d1.Id = d2.Id and  d1.PLOTX = d2.PLOTX  and  d1.PLOTY = d2.PLOTY 
    AND d1.TreeHeight<= d2.TreeHeight GROUP BY  d1.Id,d1.PLOTX,d1.PLOTY,d1.TreeHeight HAVING COUNT(d2.Id)<=3 ORDER BY d1.TreeHeight desc """)
    #分组计算前3最高树的平均值为优势高数据
    df_HT = pysqldf(""" SELECT  Id,PLOTX,PLOTY,AVG(TreeHeight) AS HT from df_HT_LIST GROUP BY  Id,PLOTX,PLOTY """)

    #将平均高几个指标进行合并
    df_HT = pysqldf(""" SELECT  df.Id,df.PLOTX,df.PLOTY,H1,H2,H3,N ,S ,HT from df join df_HT on df.Id = df_HT.Id and  df.PLOTX = df_HT.PLOTX  and  df.PLOTY = df_HT.PLOTY  """)

    # 创建保存的csv名及位置
    subplot_name = str(subcsv_name) + ".csv"
    subplot_path = os.path.join(HTplot_directory, subplot_name)
    df_HT.to_csv(subplot_path)