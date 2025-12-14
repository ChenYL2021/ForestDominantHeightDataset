# -*- coding: utf-8 -*-
# @Time    : 2023/11/30 20:40
# @Author  : ChenYuling
# @FileName: PLOTxyFunction.py
# @Software: PyCharm
# @Describe：数据处理部分：关于整理的chm的1km数据集，读取tif数据，获取中心点坐标（WGS84）,计算样地获取的树高一系列变量
###############################################################################################################################
#第1阶段，对chm的tif数据批量处理提取样地plot数据中坐标
###############################################################################################################################
#%%
import os
from osgeo import gdal
from osgeo import gdal
from pyproj import Proj, transform
import pandas as pd
import warnings
# 禁止特定类型的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
#获取tif中心点坐标
def get_tif_center_coordinates(tif_file):
    # 打开 TIF 文件
    dataset = gdal.Open(tif_file, gdal.GA_ReadOnly)
    # 获取栅格的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    # 获取栅格的地理转换信息
    transform = dataset.GetGeoTransform()
    # 计算中心点的像素坐标
    center_x = width // 2
    center_y = height // 2
    # 根据地理转换信息计算中心点的地理坐标
    center_lon = transform[0] + center_x * transform[1] + center_y * transform[2]
    center_lat = transform[3] + center_x * transform[4] + center_y * transform[5]
    # 关闭数据集
    dataset = None
    return center_lon, center_lat

#将坐标转为WGS84的坐标值

def convert_utm_to_wgs84(utm_east, utm_north):
    # 定义 UTM Zone 52N 坐标系和 WGS84 坐标系
    utm_proj = Proj(init='EPSG:32647')  # UTM Zone 52N 的 EPSG 代码为 32652 ---------------------------------------------->gaidongchu
    wgs84_proj = Proj(init='EPSG:4326')  # WGS84 坐标系的 EPSG 代码为 4326
    # 进行坐标转换
    lon, lat = transform(utm_proj, wgs84_proj, utm_east, utm_north)

    return lon, lat

#%%
#循环批量处理数据
# 文件夹路径
folder_path = r"F:\DB\TH\TrainDATA\ZONE47\tif"  # 替换为实际的文件夹路径----------------------------------------------------------->gaidongchu
# 遍历文件夹中的 TIF 文件
tif_files = [file for file in os.listdir(folder_path) if file.endswith('.tif') or file.endswith('.TIF')]
# 存储中心点坐标的列表
center_points = []

# 循环处理每个 TIF 文件
for tif_file in tif_files:
    # 构建 TIF 文件的完整路径
    tif_path = os.path.join(folder_path, tif_file)
    subtif_name = os.path.basename(tif_path).split('_')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
    if ".TIF" in subtif_name:
        modified_name = subtif_name.replace(".TIF", "")
    elif ".tif" in subtif_name:
        modified_name = subtif_name.replace(".tif", "")
    else:
        modified_name = subtif_name
    center_lon, center_lat = get_tif_center_coordinates(tif_path)# 获取 TIF 文件的中心点坐标
    # 转换为 WGS84 坐标系
    lon, lat = convert_utm_to_wgs84(center_lon, center_lat)
    # 将中心点坐标添加到列表中
    center_points.append((modified_name, lon, lat))

# 创建数据框
df = pd.DataFrame(center_points, columns=['name', 'center_lon', 'center_lat'])

#%% 保存数据框为 CSV 文件
output_file = "F:\DB\TH\TrainDATA\ZONE47.csv"  # 输出文件路径------------------------------------------------------------->gaidongchu
df.to_csv(output_file, index=False)

###############################################################################################################################
#第2阶段，对每木检尺数据批量处理成样地plot数据
###############################################################################################################################

#%%
import os
import pandas as pd

csv_directory = r"G:\lidarDATA\zone49\plot_1"  #----------------------------------------------------------------------------->gaidongchu
# 存储样地属性的列表
plot_THs = []
#%树高相关系列值计算计算
sub_tiff_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]
for subcsv_file in sub_tiff_files:
    subcsv_path = os.path.join(csv_directory, subcsv_file)
    subcsv_name = os.path.basename(subcsv_path).split('.')[0]
    print(subcsv_name)
    data = pd.read_csv(subcsv_path, sep=',')
    # data.columns = ['TreeID', 'X', 'Y','z', 'TreeHeight','CW', 'CW_SN', 'CW_EW', 'S', 'v']
    data.columns = ['TreeID', 'X', 'Y', 'TreeHeight', 'CW', 'CW_SN', 'CW_EW', 'S', 'v']
    if(len(data)==0):
        continue
    #林分平均高计算,encoding = 'gb2312'
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
output_file = "G:\lidarDATA\zone49\Z49_7.csv"  # 输出文件路径-------------------------------------------------------------------->gaidongchu
dfH.to_csv(output_file, index=False)



###############################################################################################################################
#第3阶段，将树高系列数据与坐标点数据进行合并
###############################################################################################################################
#%%
import pandas as pd

csv_directory1 = r"G:\lidarDATA\zone49\Z49.csv" ######---------------------------------------------------------------------->gaidongchu
csv_directory2 = r"G:\lidarDATA\zone49\SZ49.csv" ######------------------------------------------------------------------->gaidongchu
xy = pd.read_csv(csv_directory2, sep=',',encoding = 'gb2312')
H = pd.read_csv(csv_directory1, sep=',',encoding = 'gb2312')
merged_xyH = pd.merge(xy, H, on='name', how='inner') ######-------------------------------------------------------------->gaidongchu

#%%hebing
merged_df = pd.concat([merged_xyH44,merged_xyH46,merged_xyH47, merged_xyH48, merged_xyH49,merged_xyH50, merged_xyH51, merged_xyH52], axis=0)

#%%
output_file = "G:\lidarDATA\zone49\zone49.csv"  # 输出文件路径-------------------------------------------------------------------->gaidongchu
merged_xyH.to_csv(output_file, index=False)

#############################################*************************##################################################

#%%
csv_44 = pd.read_csv(r"F:\DB\TH\TrainDATA\Z44.csv", sep=',',encoding = 'gb2312')
csv_46 = pd.read_csv(r"F:\DB\TH\TrainDATA\Z46.csv", sep=',')#
csv_47 = pd.read_csv(r"F:\DB\TH\TrainDATA\Z47.csv", sep=',')
csv_48 = pd.read_csv(r"F:\DB\TH\TrainDATA\Z48.csv", sep=',')
csv_49 = pd.read_csv(r"F:\DB\TH\TrainDATA\Z49.csv", sep=',')
csv_50 = pd.read_csv(r"F:\DB\TH\TrainDATA\Z50.csv", sep=',')
csv_51 = pd.read_csv(r"F:\DB\TH\TrainDATA\Z51.csv", sep=',')
csv_52 = pd.read_csv(r"F:\DB\TH\TrainDATA\Z52.csv", sep=',')
CHM_data = pd.read_csv(r"G:\TH\Ydata\CHMdata.csv", sep=',',encoding = 'gb2312')

CHM_df = pd.concat([csv_44,csv_46,csv_47, csv_48, csv_49,csv_50, csv_51, csv_52], axis=0)
CHM_df1 = CHM_df[['name', 'H4']]
#%%
CHM_dat = pd.merge(CHM_data, CHM_df1, on='name', how='left') ######-------------------------------------------------------------->gaidongchu
