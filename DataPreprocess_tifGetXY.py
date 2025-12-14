# -*- coding: utf-8 -*-
# @Time    : 2024/3/20 15:37
# @Author  : ChenYuling
# @FileName: tifGetXY.py
# @Software: PyCharm
# @Describe：循环获取小tif的XY坐标数据
#%%
import pyproj
from osgeo import gdal
import warnings
import os
import pandas as pd
warnings.filterwarnings("ignore")#忽略一些版本不兼容等警告


#获取栅格中心点坐标
def get_raster_center_coordinates(raster_file):
    # 打开栅格文件
    ds = gdal.Open(raster_file)
    if ds is None:
        print("无法打开栅格文件")
        return None
    # 获取栅格的地理转换信息
    transform = ds.GetGeoTransform()
    # 获取栅格的行列数
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    # 计算栅格的中心点坐标
    x_center = transform[0] + (cols * transform[1]) / 2
    y_center = transform[3] + (rows * transform[5]) / 2
    # 关闭栅格文件
    ds = None
    return x_center, y_center

#将中心点坐标转为相应条度带坐标值
def convert_to_wgs84_utm(zone_number, easting, northing):
    utm_zone = f"+proj=utm +zone={zone_number} +datum=WGS84 +units=m +no_defs"
    wgs84 = pyproj.Proj(init='epsg:4326')  # WGS 84坐标系
    utm = pyproj.Proj(utm_zone)  # UTM投影坐标系
    lon, lat = pyproj.transform(utm, wgs84, easting, northing)
    return lon, lat

# main
# 初始化变量
tif_directory = r"F:\DB\TH\TrainDATA\CHMVersion1.2\data\Z49" #保存的原始CHM.tif文件夹位置#########################################条度带信息更改
# 存储样地属性的列表
plot_XY = []
# 获取文件夹中的tif文件路径列表
sub_tiff_files = [file for file in os.listdir(tif_directory) if file.endswith('.tif')]
for subtif_file in sub_tiff_files:#该循环结束后，sub_shapefile_df数据框保存该小shp块下对应自变量列数据
    subtif_path = os.path.join(tif_directory, subtif_file)
    subtif_name = os.path.basename(subtif_path).split('.')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
    center_x, center_y = get_raster_center_coordinates(subtif_path)
    # 将UTM投影坐标转换为WGS 1984（4*条带）坐标
    center_lon, center_lat = convert_to_wgs84_utm(49, center_x, center_y)  ###################################################条度带信息更改
    print(f"栅格文件的中心点WGS 1984坐标为：({center_lon}, {center_lat})")
    plot_XY.append((subtif_name, center_lon, center_lat))

    print("CreateFishnet:",subtif_name,"over!")

# 创建数据框
dfxy = pd.DataFrame(plot_XY, columns=['name', 'center_lon', 'center_lat'])
# % 保存数据框为 CSV 文件
output_file = "F:\DB\TH\TrainDATA\CHMVersion1.2\data\Z49.csv"  # 输出文件路径-------------------------------------------------------------------->信息更改
dfxy.to_csv(output_file, index=False)

