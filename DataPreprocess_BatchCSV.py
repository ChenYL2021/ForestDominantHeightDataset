# -*- coding: utf-8 -*-
# @Time    : 2023/10/26 21:02
# @Author  : ChenYuling
# @FileName: BatchCSV.py
# @Software: PyCharm
# @Describe：数据处理部分：实现将文件夹下所有csv文件合并成一个文件

#%%各区域汇总csv合并成总scv
import os
import pandas as pd
csv_directory = r"G:\dbfPLOT\csvPLOT"
BigPlot = r"G:\dbfPLOT\csvPLOT\all"
# 创建输出目录（如果不存在）
if not os.path.exists(BigPlot):
    os.makedirs(BigPlot)

# 获取文件夹中的tif文件路径列表
sub_tiff_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]
df = pd.DataFrame()
for subcsv_file in sub_tiff_files:
    subcsv_path = os.path.join(csv_directory, subcsv_file)
    subcsv_name = os.path.basename(subcsv_path).split('.')[0]
    data = pd.read_csv(subcsv_path, sep=',')

    df = pd.concat([df, data])

# 创建保存的csv名及位置
subplot_name = "ALLplots.csv"
subplot_path = os.path.join(BigPlot, subplot_name)
df.to_csv(subplot_path, index=None)


#%%所有区域文件夹合并，删除明显异常值S<800
import os
import pandas as pd
csv_directory = r"F:\DB\CHM\ALLplot"
BigPlot = r"F:\DB\CHM\ALLplot"
# 创建输出目录（如果不存在）
if not os.path.exists(BigPlot):
    os.makedirs(BigPlot)

# 获取文件夹中的tif文件路径列表
sub_tiff_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]
df = pd.DataFrame()
for subcsv_file in sub_tiff_files:
    subcsv_path = os.path.join(csv_directory, subcsv_file)
    subcsv_name = os.path.basename(subcsv_path).split('.')[0]
    data = pd.read_csv(subcsv_path, sep=',')

    df = pd.concat([df, data])

# 删除明显异常值S<=800
df1 = df[['Id','PLOTX','PLOTY','H1','H2','H3','N','S']]
df2 = df1[df1['S'] > 450]
# 创建保存的csv名及位置
subplot_name = "XYplots.csv"
subplot_path = os.path.join(BigPlot, subplot_name)
df2.to_csv(subplot_path)



#%%实现获取各个文件名称
import os
import pandas as pd
csv_directory = r"H:\样本Lidar数据\zone_52"
BigPlot = r"H:\样本Lidar数据"
# 创建输出目录（如果不存在）
if not os.path.exists(BigPlot):
    os.makedirs(BigPlot)

# 获取文件夹中的tif文件路径列表
sub_tiff_files = [file for file in os.listdir(csv_directory) if file.endswith('.las') or file.endswith('.LiData') ]
dfLIST = []
for subcsv_file in sub_tiff_files:
    subcsv_path = os.path.join(csv_directory, subcsv_file)
    subcsv_name = os.path.basename(subcsv_path).split('.')[0]
    dfLIST.append(subcsv_name)

# 将列表转换为数据框中的一列
df = pd.DataFrame({'NameLAS': pd.Series(dfLIST)})
df["file"] = 'zone_52'
# 创建保存的csv名及位置
# 创建保存的csv名及位置
subplot_name = "NameLAS52.csv"
subplot_path = os.path.join(BigPlot, subplot_name)
df.to_csv(subplot_path)

#%%实现将两个csv合并，获得训练数据（未清洗）
import os
from pandasql import sqldf
pysqldf = lambda q:sqldf(q,globals())
df_XYH = pd.read_csv('F:\DB\CHM\ALLplot\csv\P312741.csv')#wgs1984坐标和平均高密度参数
df_HT = pd.read_csv('F:\DB\CHM\ALLplot\csv\ALLplots.csv')#优势高参数

df_H = pysqldf(""" SELECT  df_XYH.Id,df_XYH.PLOTX,df_XYH.PLOTY,df_XYH.H1,df_XYH.H2,df_XYH.H3,df_XYH.N,df_XYH.S, HT from df_XYH join df_HT on
                    df_XYH.Id = df_HT.Id AND df_XYH.H1 = df_HT.H1 AND df_XYH.H2 = df_HT.H2 AND df_XYH.H3 = df_HT.H3 AND df_XYH.N = df_HT.N AND df_XYH.S = df_HT.S  """)


#%%将CSV转为shp
import arcpy
from arcpy import env
import os
import arcgisscripting
gp = arcgisscripting.create()
spatial_ref = arcpy.SpatialReference(4326)#设置为WGS84坐标系
env.workspace = r'F:\DB\CHM\dongbeidayangdi\HTplot'#原始文本文件所在地址
pathout = r'F:\DB\CHM\dongbeidayangdi\HTplot\shp'#输出结果地址
# 创建输出目录（如果不存在）
if not os.path.exists(pathout):
    os.makedirs(pathout)
x_corrods = 'PLOTX'#设置关键字
y_corrods = 'PLOTY'
z = 'Id'
try:
    for file1 in arcpy.ListFiles("*.csv"):
        print(file1)
        info = os.path.basename(file1).split('.')[0]
        intable = file1
        outlayer = info
        print('outlayer', outlayer)
        gp.MakeXYEventLayer_management(intable, x_corrods, y_corrods, outlayer, spatial_ref, z)
        print('MakeXYEventLayer over')
        gp.FeatureClassToShapefile_conversion(outlayer, pathout)
        print('ToShapefile over')
except:
    print(gp.GetMessages())


#%%
# 版权声明：本文为CSDN博主「_Jinyuan」的原创文章，遵循CC
# 4.0
# BY - SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https: // blog.csdn.net / GggggYyyyy111 / article / details / 119617936
import arcpy
import os

file_path = r'F:\DB\CHM\dongbeidayangdi\HTplot\shp'
file_list = os.listdir(file_path)

for file in file_list:
    # shapefile 含有很多其他附属文件，这一步单独将「.shp 」筛选出来
    if 'shp' in file:
        file_dir = os.path.join(file_path, file)
        # 定义坐标系为 WGS84（代码4326）
        sr = arcpy.SpatialReference(4326)
        arcpy.DefineProjection_management(file_dir, sr)
        print(file + u'投影成功')
    else:
        print(u'跳过')
