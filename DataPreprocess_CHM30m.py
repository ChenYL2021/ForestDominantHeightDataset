# -*- coding: utf-8 -*-
# @Time    : 2024/3/18 10:17
# @Author  : ChenYuling
# @FileName: CHM30m.py
# @Software: PyCharm
#%% @Describe：数据处理部分：实现大CHM切割成多个小tif（30m）
import os
from osgeo import gdal, gdalconst

def split_tiff(input_tiff, output_dir, tile_size=15):
    # 打开输入的TIFF文件
    dataset = gdal.Open(input_tiff, gdalconst.GA_ReadOnly)
    if dataset is None:
        print("无法打开输入的TIFF文件")
        return

    # 获取TIFF文件的基本信息
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # 计算裁剪的行数和列数
    rows = int((height + tile_size - 1) / tile_size)
    cols = int((width + tile_size - 1) / tile_size)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 逐个裁剪TIFF文件
    for i in range(rows):
        for j in range(cols):
            # 计算当前裁剪的起始像素坐标
            x_offset = j * tile_size
            y_offset = i * tile_size

            subtif_name = os.path.basename(input_tiff).split('.')[0]
            sub_name = str(subtif_name) + f"_{i}_{j}.tif"
            output_filename = os.path.join(output_dir, sub_name)

            # 使用gdal.Translate裁剪TIFF文件
            gdal.Translate(output_filename, dataset, srcWin=[x_offset, y_offset, tile_size, tile_size])

            # 设置输出小块的地理参考信息
            output_dataset = gdal.Open(output_filename, gdalconst.GA_Update)
            output_dataset.SetGeoTransform((geotransform[0] + x_offset * geotransform[1], geotransform[1], 0,
                                            geotransform[3] + y_offset * geotransform[5], 0, geotransform[5]))
            output_dataset.SetProjection(projection)
            output_dataset = None

    # 关闭输入TIFF文件
    dataset = None



#%% 用法
tif_directory =  r"F:\DB\TH\TrainDATA\CHMVersion1.2\data\tif50"
output_dir = r"F:\DB\TH\TrainDATA\CHMVersion1.2\Z47"

# 获取文件夹中的tif文件路径列表
sub_tiff_files = [file for file in os.listdir(tif_directory) if file.endswith('.tif')]
for subtif_file in sub_tiff_files:#该循环结束后，sub_shapefile_df数据框保存该小shp块下对应自变量列数据
    input_tiff = os.path.join(tif_directory, subtif_file)
    subtif_name = os.path.basename(input_tiff).split('.')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
    print(subtif_name)
    # 创建保存的tif名及位置
    split_tiff(input_tiff, output_dir)
