# -*- coding: utf-8 -*-
# @Time    : 2023/12/1 13:15
# @Author  : ChenYuling
# @FileName: PLOTlidar.py
# @Software: PyCharm
# @Describe：数据处理部分：关于整理的lidar的1km数据集，读取las数据，获取中心点坐标（WGS84）,计算样地获取的树高一系列变量
########################################################################################################################
##############################第1阶段，对las数据批量处理提取样地plot数据中坐标#################################################
########################################################################################################################
import os
import laspy
import numpy as np
from scipy import spatial
import pandas as pd
import open3d as o3d
import math
from osgeo import osr
from scipy.interpolate import UnivariateSpline, CubicSpline
from osgeo import gdal

def get_all_filenames(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            name_without_extension, _ = os.path.splitext(filename)
            filenames.append(name_without_extension)
    return filenames


def print_axis(las_file_name):
    # las文件位置 示例数据为机载数据 但是算法通用
    lasfile = las_file_name
    # 打开las文件
    inFile = laspy.read(lasfile)
    # DEM
    x_max, y_max = inFile.header.max[0:2]
    x_min, y_min = inFile.header.min[0:2]
    x_now=(x_max+x_min)/2
    y_now=(y_max+y_min)/2
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromProj4("+proj=utm +zone=49 +datum=WGS84 +units=m +no_defs")###---------------------------------------------->gaidongchu
    geosc1=outRasterSRS.CloneGeogCS()
    cor_tran=osr.CoordinateTransformation(outRasterSRS,geosc1)
    coords=cor_tran.TransformPoint(x_now,y_now)
    return coords

#%%
# 存储中心点坐标的列表
center_points = []

file_path = 'G:\\zone49张艺轩\\las\\'   # 输出文件路径------------------------------------------------------------------------>gaidongchu
file_name = get_all_filenames(file_path)

for name in file_name:
    print_axis(file_path + name + ".las")
    coords = print_axis(file_path+name+".las")
    print("start",name)
    if "_转换为Las" in name:
        modified_name = name.replace("_转换为Las", "")
    else:
        modified_name = name
    # 将中心点坐标添加到列表中
    center_points.append((modified_name, coords[0],coords[1]))

# 创建数据框
df = pd.DataFrame(center_points, columns=['name', 'center_lon', 'center_lat'])

#%% 保存数据框为 CSV 文件
output_file = "G:\lidarDATA\ZONE49_ZYX.csv"  # 输出文件路径---------------------------------------------------------------------->gaidongchu
df.to_csv(output_file, index=False)

###############################################################################################################################
#第2阶段，对每木检尺数据批量处理成样地plot数据
###############################################################################################################################

#%%
import os
import pandas as pd

csv_directory = r"G:\lidarDATA\zone49\plot100"  #----------------------------------------------------------------------------->gaidongchu
# 存储样地属性的列表
plot_THs = []
#%树高相关系列值计算计算
sub_tiff_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]
for subcsv_file in sub_tiff_files:
    subcsv_path = os.path.join(csv_directory, subcsv_file)
    subcsv_name = os.path.basename(subcsv_path).split('.')[0]
    data = pd.read_csv(subcsv_path, sep=',')#, encoding = 'gb2312'
    if data.shape[1]==10:
        data.columns = ['TreeID', 'X', 'Y','TreeHeight','DBH','CW', 'CW_SN', 'CW_EW', 'S', 'V']#
    elif data.shape[1]==9:
        data.columns = ['TreeID', 'X', 'Y', 'TreeHeight','CW', 'CW_SN', 'CW_EW', 'S', 'V']  #
    elif data.shape[1]==11:
        data.columns = ['TreeID', 'X', 'Y','Z','TreeHeight','DBH','CW', 'CW_SN', 'CW_EW', 'S', 'V']  #
    # data.columns = ['PLOT', 'DBH', 'TreeHeight', 'CW', 'CW_SN', 'CW_EW', 'S','V','biomass'] #机载
    #林分平均高计算
    # df = pysqldf(
    #     """ SELECT  PLOT,SUM(biomass) AS B, AVG(TreeHeight) AS Hmean,AVG(CW) AS CWmean from data GROUP BY PLOT""")
    # df1 = pd.merge(data, df, on='PLOT', how='inner')
    # df2 = pd.read_csv(subcsv_path, sep=',')#, encoding = 'gb2312' 读坐标
    # df3 = pd.merge(df1, df2, on='PLOT', how='inner')
    if len(data)>0:
        H1 = data['TreeHeight'].mean()
        H2 = (data['TreeHeight'] * data['TreeHeight']).sum() / data['TreeHeight'].sum()
        H3 = (data['TreeHeight'] * data['TreeHeight'] * data['TreeHeight']).sum()/(data['TreeHeight'] * data['TreeHeight']).sum()
        H4 = (data['TreeHeight'] * data['CW']).sum() / data['CW'].sum()
        N = len(data)
        CW_mean = data['CW'].mean()
        CWSN_mean = data['CW_SN'].mean()
        CWEW_mean = data['CW_EW'].mean()
        CWArea_sum = data['S'].sum()
        CWVolume_sum = data['V'].sum()
        #筛选出每组前3条最大树高数据集
        dataTOP3 = (data.sort_values(by='TreeHeight', ascending=False)).head(3)
        #分组计算前3最高树的平均值为优势高数据
        HT = dataTOP3['TreeHeight'].mean()
        #将平均高几个指标进行合并
        plot_THs.append((subcsv_name, H1, H2,H3,H4,N,CW_mean,CWSN_mean,CWEW_mean,CWVolume_sum,CWArea_sum,HT))
    else:
        continue

# 创建数据框
dfH = pd.DataFrame(plot_THs, columns=['plot', 'H1', 'H2', 'H3','H4', 'N', 'CW_mean', 'CWSN_mean', 'CWEW_mean', 'CWVolume_sum','CWArea_sum', 'HT'])
# %保存数据框为 CSV 文件
output_file = "G:\lidarDATA\zone49\Z49_100.csv"  # 输出文件路径-------------------------------------------------------------------->gaidongchu
dfH.to_csv(output_file,  index=False,encoding='utf-8-sig')

#%%
#######################################%%各区域汇总csv合并成总scv
import os
import pandas as pd
csv_directory = r"G:\lidarDATA\zone49\spatial100"
BigPlot = r"G:\lidarDATA\zone49"
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
subplot_name = "SZ49_100.csv"
subplot_path = os.path.join(BigPlot, subplot_name)
df.to_csv(subplot_path,  index=False,encoding='utf-8-sig')



#########################################################################################
#第3阶段，将树高系列数据与坐标点数据进行合并
###############################################################################################################################
#%%
import pandas as pd
csv_directory1 = r"G:\lidarDATA\zone49\Z49_100.csv"
csv_directory2 = r"G:\lidarDATA\zone49\SZ49_100.csv"
xy = pd.read_csv(csv_directory2, sep=',')
Hdat = pd.read_csv(csv_directory1, sep=',') #,encoding = 'gb2312'

merged_xyH = pd.merge(xy, Hdat, on='plot', how='inner')
merged_xyH["id"] = merged_xyH.reset_index().index+1

merged_xyH.to_csv("G:\lidarDATA\zone49\zone49_100.csv",index=False,encoding='utf-8-sig')

#%%hebing
ZONE46 = pd.read_csv("G:\lidarDATA\zone46\zone46.csv", sep=',')
ZONE47 = pd.read_csv("G:\lidarDATA\zone47\zone47.csv", sep=',')
ZONE48 = pd.read_csv("G:\lidarDATA\zone48\zone48.csv", sep=',')
ZONE49 = pd.read_csv("G:\lidarDATA\zone49\zone49.csv", sep=',')
ZONE50 = pd.read_csv("G:\lidarDATA\zone50\zone50.csv", sep=',')
ZONE52 = pd.read_csv("G:\lidarDATA\zone52\zone52.csv", sep=',')

merged_df = pd.concat([ZONE46, ZONE47, ZONE48,ZONE49,ZONE50,ZONE52], axis=0)
merged_df["PLOTID"] = merged_df.reset_index().index+1
#%%
output_file = "G:\TH\TrainDATA30m.csv"  # 输出文件路径-------------------------------------------------------------------->gaidongchu
merged_df.to_csv(output_file,index=False,encoding='utf-8-sig')



